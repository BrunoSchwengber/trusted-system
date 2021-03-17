timestamp() {
  date +"%Y-%m-%d %H:%M:%S,%3N"
}

d=$(date +'%T,%3N')
echo $d

python3 timestamp-test.py $d
