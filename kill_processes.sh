# Kill all processes listed in pids.txt
while read pid; do
  kill $pid
done < pids.txt

# Optionally, remove the pids.txt file
rm pids.txt