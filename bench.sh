#!/bin/bash


function bench1() {
	# Clean model name for filename by replacing ':' with '-'
	MODEL="$1"
	CLEAN_MODEL="$(echo "$MODEL" | tr ':' '-' | tr '/' '_')"
	TIMESTAMP="$(date +%Y-%m-%d-%H-%M)"
	LOGFILE="benchmark-$CLEAN_MODEL.log"

	# Start benchmark
	echo '' > $LOGFILE
	echo "Starting benchmark for $MODEL at $TIMESTAMP..." | tee -a $LOGFILE

	# Show model information
	echo "Model information:" >> $LOGFILE
	ollama show "$MODEL" >> $LOGFILE 2>&1
	MODEL_MAX_CTX="$(ollama show $MODEL | grep 'context length' | awk '{print $3}')"
	
	RUN=1
	
	# Run this test against multiple context window sizes to provide metrics about which would be the best.
	for CTX in 8192 16384 24576 32768 40960 49152 57344 65536 73728 81920 90112 98304 106496 114688 122880 131072 139264 147456 155648 163840 172032 180224 188416 196608 204800; do
	
		if [ -n "$MODEL_MAX_CTX" ] && [ $CTX -gt $MODEL_MAX_CTX ]; then
			# Only bench up to the maximum context size of the model.
			# This is particularly useful for small models that have a 32k context size.
			break
		fi
	
		if [ $RUN -ne 1 ]; then
			break
		fi

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
}' | tee -a $LOGFILE

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
			echo "Memory used: $(numfmt --to=iec-i --suffix=B $MEM)" | tee -a $LOGFILE
		else
			echo "Unable to determine memory used!  Please ensure you have nvidia-smi or rocm-smi installed." | tee -a $LOGFILE
		fi
		
		# Check if the model started to access system memory...
		# If it does, we should stop the benchmarks to prevent system stability issues and
		# because any tests past this one will simply result in continually diminishing results.
		if [ -n "$(ollama ps | grep 'CPU/GPU')" ]; then
			echo '!!WARNING!! Model started to expand past video memory; halting run for this model' | tee -a $LOGFILE
			RUN=0
		fi

		# Clean up
		echo "Cleaning up..."
		ollama stop "$MODEL"
	done

	echo "Benchmark completed. Results in $LOGFILE"
}


# Check if MODEL is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <MODEL_NAME>"
    echo ""
    echo "Available Models:"
    ollama list | awk '{print $1}' | tail -n +2 | sort
    exit 1
fi

if [ "$1" == "all" ]; then
	for MODEL in $(ollama list | awk '{print $1}' | tail -n +2 | sort); do
		bench1 "$MODEL"
	done
else
	bench1 "$1"
fi

