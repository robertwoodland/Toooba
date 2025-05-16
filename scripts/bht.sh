#!/bin/bash

PRED="Bht"
UPPER="BHT"
LOWER="bht"
PRED_FILE="src_Core/RISCY_OOO/procs/lib/$PRED.bsv"
FILE_NAME="./$PRED.bsv"
PROC_CONFIG="src_Core/RISCY_OOO/procs/RV64G_OOO/ProcConfig.bsv"
BUILD_DIR="builds/RV64ACDFIMSU_Toooba_bluesim"
COMMAND="./exe_HW_sim"
GREP_CMD='grep instret:188799'
TOPLEVEL="$(pwd)"

ENTRIES=(512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)

# Aight so 
# 0. Pick predictor
# 1. Change perceptron file to new budget
# 2. make compile
# 3. make simulate
# 4. run exe, pipe into file
# 5. grep for finished
# 6. stop exe

# 0:
sed -i 's|`define\s\+DIR_PRED_BHT|`define DIR_PRED_'"${UPPER}|" "$PROC_CONFIG"
sed -i 's|`define\s\+DIR_PRED_TOUR|`define DIR_PRED_'"${UPPER}|" "$PROC_CONFIG"
sed -i 's|`define\s\+DIR_PRED_PERCEPTRON|`define DIR_PRED_'"${UPPER}|" "$PROC_CONFIG"

# 1:
for i in {0..11}; do
    SIZE=$(python3 ./scripts/size.py $LOWER ${ENTRIES[i]})
    echo "Updating ${PRED} to size: ${SIZE}, entries: ${ENTRIES[i]}"
    cd "$(dirname "$PRED_FILE")" || exit 1
    OUTPUT_FILE="${SIZE}_${ENTRIES[i]}.txt"

    # Modify params
    sed -i "s/typedef\s\+\w\+\s\+BhtEntries;/typedef ${ENTRIES[i]} BhtEntries;/" "$FILE_NAME"

    echo "Changing to top"
    # Change back to top
    cd "-" || exit 1

    # 2,3:
    cd "$BUILD_DIR" || exit 1
    echo "Compiling $PRED"
    for attempt in {1..15}; do
        if make all; then
            break
        elif [ "$attempt" -eq 15 ]; then
            echo "make all failed after 15 attempts"
            exit 1
        else
            echo "make all failed, retrying ($attempt/15)..."
        fi
    done

    echo "Executable ready, running $PRED"

    # 4:
    echo "Running $PRED simulator"
    OUTFILE="${TOPLEVEL}/results/${PRED}/${OUTPUT_FILE}"
    $COMMAND > "${OUTFILE}" &
    CMD_PID="$!"
    trap "kill ${CMD_PID}" EXIT # Ensure CoreMark killed on exit

    echo "Changing back to top"
    cd "$TOPLEVEL" || exit 1

    echo "It's running, I'm sleeping (JIC). PID: ${CMD_PID}"
    # sleep 30 # Sleep for 30 seconds
    # # 5:
    echo "Grep for results"
    for attempt in {1..300}; do
        if $GREP_CMD "${OUTFILE}" > /dev/null; then 
            echo "Match found in ${OUTFILE}";
            # 6:
            sleep 10 # Sleep for 10 seconds to ensure the file is written
            echo "Killing simulator"
            kill $CMD_PID # Kill CoreMark
            break;
        elif [ "$attempt" -eq 300 ]; then
            echo "CoreMark failed after 15 minutes";
            exit 1; # Will kill CoreMark
        else
            echo $GREP_CMD "${OUTFILE}";
            echo "No match found, retrying ($attempt/300)..."
            sleep 3;
        fi
    done
done
