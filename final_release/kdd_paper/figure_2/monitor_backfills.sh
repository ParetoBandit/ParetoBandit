#!/bin/bash
# Periodic health monitor for all backfill processes

while true; do
    clear
    echo "=== BACKFILL HEALTH MONITOR ==="
    echo "Time: $(date)"
    echo ""
    
    # Check running processes
    echo "Active Processes:"
    ps aux | grep "backfill" | grep python3 | grep -v grep | awk '{print "  "$12" (PID "$2")"}'
    echo ""
    
    # Check each log
    for name in gpt5:70 gemini:69 claude:151 o3:14 gpt51:11 comprehensive:13513; do
        model=$(echo $name | cut -d: -f1)
        total=$(echo $name | cut -d: -f2)
        logfile="/tmp/${model}_backfill.log"
        
        if [ -f "$logfile" ]; then
            # Get last progress line
            progress=$(grep -E "^\[" "$logfile" | tail -1)
            errors=$(grep -c "✗" "$logfile" 2>/dev/null || echo 0)
            successes=$(grep -c "✓" "$logfile" 2>/dev/null || echo 0)
            
            if [ -n "$progress" ]; then
                echo "[$model] $progress"
                echo "         Success: $successes | Errors: $errors"
            else
                echo "[$model] Starting... (Total: $total)"
            fi
        else
            echo "[$model] Log not found"
        fi
        echo ""
    done
    
    echo "Next update in 60 seconds (Ctrl+C to stop)..."
    sleep 60
done
