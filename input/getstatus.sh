qstat -u jklymak
grep -E 'advcfl_wvel_max|time_seconds|dynstat_uvel_max' ../results/lee3dfull01/_Model/input/STDOUT.0000 | tail -n 9
ls -alt ../results/lee3dfull01/0000/T*.data | head -n 3
