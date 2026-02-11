#!/bin/bash
# Restart training with Apple MPS GPU acceleration

echo "Stopping current CPU training..."
pkill -f train_on_ett.py
sleep 2

echo "Starting training with MPS GPU acceleration..."
cd /Users/pranavtripathi/Desktop/DNLP/DNLPproject
nohup python -u train_on_ett.py > training_output.log 2>&1 &

echo "Training started! Process ID: $!"
echo ""
echo "Monitor with: tail -f training_output.log"
echo "Check status with: ps aux | grep train_on_ett.py"
