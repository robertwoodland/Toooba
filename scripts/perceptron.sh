#!/bin/bash

PRED="Perceptron"
UPPER="PERCEPTRON"
LOWER="perceptron"
PRED_FILE="src_Core/RISCY_OOO/procs/lib/$PRED.bsv"
FILE_NAME="./$PRED.bsv"
PROC_CONFIG="src_Core/RISCY_OOO/procs/RV64G_OOO/ProcConfig.bsv"
BUILD_DIR="builds/RV64ACDFIMSU_Toooba_bluesim"
COMMAND="./exe_HW_sim"
GREP_CMD='grep instret:188799'
TOPLEVEL="$(pwd)"

# New test rig gcc:
LOCAL=(2 2 2 5 5 10 10 11 12 14 15 18)
GLOBAL=(8 10 23 25 31 34 34 36 51 71 115 155)
COUNT=(11 19 19 33 55 91 182 341 500 750 1000 1500)

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
    SIZE=$(python3 ./scripts/size.py $LOWER ${LOCAL[i]} ${GLOBAL[i]} ${COUNT[i]})
    echo "Updating ${PRED} to size: ${SIZE}, local: ${LOCAL[i]}, global: ${GLOBAL[i]}, count: ${COUNT[i]}"
    cd "$(dirname "$PRED_FILE")" || exit 1
    OUTPUT_FILE="${SIZE}_${LOCAL[i]}_${GLOBAL[i]}_${COUNT[i]}.txt"
    
    # Modify params
    sed -i "s/typedef\s\+\w\+\s\+PerceptronEntries;/typedef ${LOCAL[i]} PerceptronEntries;/" "$FILE_NAME"
    sed -i "s/typedef\s\+\w\+\s\+PerceptronGHistEntries;/typedef ${GLOBAL[i]} PerceptronGHistEntries;/" "$FILE_NAME"
    sed -i "s/typedef\s\+\w\+\s\+PerceptronCount;/typedef ${COUNT[i]} PerceptronCount;/" "$FILE_NAME"

    echo "Changing to top"
    # Change back to top
    cd "-" || exit 1

    # 2,3:
    cd "$BUILD_DIR" || exit 1
    echo "Compiling $PRED"
    for attempt in {1..10}; do
        if make all; then
            break
        elif [ "$attempt" -eq 10 ]; then
            echo "make all failed after 10 attempts"
            exit 1
        else
            echo "make all failed, retrying ($attempt/10)..."
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
