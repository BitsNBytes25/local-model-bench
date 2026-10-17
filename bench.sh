#!/bin/bash

# Dependencies check
if [ ! command -v jq &> /dev/null ]; then
	echo "Please install jq" >&2
	exit 1
fi
if [ ! command -v nvidia-smi &> /dev/null ] && [ ! command -v rocm-smi &> /dev/null ]; then
	echo "Either nvidia-smi or rocm-smi (for nvidia/amd respectively) is required" >&2
	exit 1
fi
if [ ! command -v ollama &> /dev/null ]; then
	echo "Please install ollama" >&2
	exit 1
fi
if [ ! command -v curl &> /dev/null ]; then
	echo "Please install curl" >&2
	exit 1
fi

if [ $(ollama ps | wc -l) -gt 1 ]; then
	echo "Ollama model already loaded, refusing to run benchmark" >&2
	exit 1
fi

if [ command -v nvidia-smi &> /dev/null ]; then
	echo "Nvidia-smi is not supported yet" >&2
	echo "The commands need fixed by someone with an nvidia card." >&2
	exit 1
fi


function bench1() {
	# Clean model name for filename by replacing ':' with '-'
	MODEL="$1"
	CLEAN_MODEL="$(echo "$MODEL" | tr ':' '-' | tr '/' '_')"
	TIMESTAMP="$(date +%Y%m%d-%H%M)"
	LOGFILE="data/benchmark-$CLEAN_MODEL.log"
	RESPONSE="$(mktemp --suffix=.json)"
	DATA="data/benchmark-data.csv"
	
	# Ensure the data directory exists for storing results
	[ -d data/ ] || mkdir data/

	if command -v nvidia-smi &> /dev/null; then
		TOTAL_MEM_MB="$(nvidia-smi --id=0 --format=json --query-gpu=memory.total | jq -r '.nvidia_smi_log.gpus[].fb_memory_usage.total' | sed 's: MiB::')"
		CARD_NAME="$(nvidia-smi --id=0 --format=json --query-gpu=name | jq -r '.nvidia_smi_log.gpus[].product_name')"
	elif command -v rocm-smi &> /dev/null; then
		# {"card0": {"VRAM Total Memory (B)": "25753026560", "VRAM Total Used Memory (B)": "25718833152"}}
		TOTAL_MEM_MB="$(rocm-smi --showmeminfo vram --json --device 0 | jq -r '.[]["VRAM Total Memory (B)"] | tonumber / 1024 / 1024')"
		CARD_NAME="$(rocm-smi --showproductname --json --device 0 | jq -r '.[]["Subsystem ID"]')"
	else
		TOTAL_MEM_MB="0"
		CARD_NAME=""
	fi
	

	# Start benchmark
	echo '' > $LOGFILE
	echo "Starting benchmark for $MODEL at $TIMESTAMP..." | tee -a $LOGFILE

	# Show model information
	echo "Model information:" >> $LOGFILE
	ollama show "$MODEL" >> $LOGFILE 2>&1
	MODEL_MAX_CTX="$(ollama show $MODEL | grep 'context length' | awk '{print $3}')"
	
	RUN=1

	if [ $TOTAL_MEM_MB -le 8192 ]; then
		CTX_JUMP=1024
	elif [ $TOTAL_MEM_MB -le 16384 ]; then
		CTX_JUMP=4096
	elif [ $TOTAL_MEM_MB -le 24576 ]; then
		CTX_JUMP=8192
	else
		CTX_JUMP=16384
	fi
	CTX_SIZES=()
    for i in {1..32}; do
        CTX_SIZES+=( $(( $CTX_JUMP * i )) )
    done
	
	# Run this test against multiple context window sizes to provide metrics about which would be the best.
	for CTX in "${CTX_SIZES[@]}"; do
	
		if [ -n "$MODEL_MAX_CTX" ] && [ $CTX -gt $MODEL_MAX_CTX ]; then
			# Only bench up to the maximum context size of the model.
			# This is particularly useful for small models that have a 32k context size.
			break
		fi
	
		if [ $RUN -ne 1 ]; then
			break
		fi
		
		# Each run iteration updates the timestamp.
		TIMESTAMP="$(date +%Y%m%d-%H%M)"

		# Run the model with verbose output
		echo "" | tee -a $LOGFILE
		echo "===================================================================" | tee -a $LOGFILE
		echo "Running model at $CTX context length..." | tee -a $LOGFILE
		echo "===================================================================" | tee -a $LOGFILE
		echo "" | tee -a $LOGFILE
		curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"$MODEL\",
  \"prompt\": \"Write a 'hello world' application in Python\",
  \"stream\": false,
  \"options\": {
    \"num_ctx\": $CTX
  }
}" | jq 'del(.context) | . + {
  "tokens_per_second": (if .eval_duration > 0 then (((.eval_count / .eval_duration) * 1000000000) * 100 | round / 100) else 0 end),
  "time_to_first_token_ms": (if .prompt_eval_duration > 0 then ((.prompt_eval_duration / 1000000) * 100 | round / 100) else 0 end)
}' | tee -a $LOGFILE | tee $RESPONSE
		
		# Extract the various data points from the response; this is JSON so we can use jq
		EVAL_DURATION_MS="$(jq '(if .eval_duration > 0 then ((.eval_duration / 1000000) * 100 | round / 100) else 0 end)' < $RESPONSE)"
		TOKENS="$(jq '.eval_count' < $RESPONSE)"
		TOKENS_PER_SEC="$(jq '.tokens_per_second' < $RESPONSE)"
		

		# Verify context window
		echo "" | tee -a $LOGFILE
		echo "-------------------------------------------------------------------" | tee -a $LOGFILE
		echo "Process information:" >> $LOGFILE
		ollama ps >> $LOGFILE

		# Get GPU memory usage
		echo "" >> $LOGFILE
		echo "-------------------------------------------------------------------" >> $LOGFILE
		echo "GPU memory usage:" >> $LOGFILE
		PID="$(ps aux | grep llama-server | head -n1 | awk '{print $2}')"

		# Check if it's an NVIDIA GPU
		MEM=""
		if command -v nvidia-smi &> /dev/null; then
			if [ -n "$PID" ]; then
				MEM="$(nvidia-smi --query-gpu=memory.total --compute-process "$PID" -l 1)"
			fi
		else
			if command -v rocm-smi &> /dev/null; then
				if [ -n "$PID" ]; then
					MEM="$(rocm-smi --showpids | grep $PID | awk '{print $4}')"
				fi
			fi
		fi
		
		if [ -n "$MEM" ]; then
			MEM_MB=$(( MEM / 1024 / 1024 ))
			echo "Memory used: $(numfmt --to=iec-i --suffix=B $MEM)" | tee -a $LOGFILE
		else
			MEM_MB=""
			echo "Unable to determine memory used!  Please ensure you have nvidia-smi or rocm-smi installed." | tee -a $LOGFILE
		fi
		
		# Check if the model started to access system memory...
		# If it does, we should stop the benchmarks to prevent system stability issues and
		# because any tests past this one will simply result in continually diminishing results.
		if [ -n "$(ollama ps | grep 'CPU/GPU')" ]; then
			OFFLOAD="x"
			echo '!!WARNING!! Model started to expand past video memory; halting run for this model' | tee -a $LOGFILE
			RUN=0
		else
			OFFLOAD=""
		fi

		# Clean up
		echo "Cleaning up..."
		ollama stop "$MODEL"
		
		# Save the raw data which can be used for graphing
		if [ ! -f $DATA ]; then
			echo "Card,Date Time,Model,Context,Max Context,Memory (MB),Max Mem (MB),Tokens,Duration (ms),Tokens/sec,Offload" > $DATA
		fi
		echo "$CARD_NAME,$TIMESTAMP,$MODEL,$CTX,$MODEL_MAX_CTX,$MEM_MB,$TOTAL_MEM_MB,$TOKENS,$EVAL_DURATION_MS,$TOKENS_PER_SEC,$OFFLOAD" >> $DATA
	done

	echo "Benchmark completed. Results in $LOGFILE"
}


# Check if MODEL is provided
if [ -z "$1" ]; then
    echo "Usage: $0 [<MODEL_NAME> | all]"
    echo ""
    echo "Available Models:"
    ollama list | awk '{print $1}' | tail -n +2 | sort
    echo ""
    echo "Enter a model name to benchmark, or 'all' to bench all models listed"
    exit 1
fi

if [ "$1" == "all" ]; then
	for MODEL in $(ollama list | awk '{print $1}' | tail -n +2 | sort); do
		bench1 "$MODEL"
	done
else
	bench1 "$1"
fi
