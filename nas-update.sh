#!/bin/zsh

# this script updates 2d-discharge-nn on the remote directory

terminal-notifier -title 'rsync Backup' -subtitle 'Starting' -message 'Backup started.' -sound Glass

LOG_FILE="/Users/jarl/2d-discharge-nn/backup_log.txt"
now=$(date)

cat << EOF > $LOG_FILE
**** rsync log file for created_models ****
Started on: $now

EOF

rsync -avP --stats ~/2d-discharge-nn/ /Volumes/home/Public_HamaLab/Data/22_jarl/2d-discharge-nn >> $LOG_FILE
 
end=$(date)
printf "\nFinished on: $end" >> $LOG_FILE

# stats
transferred=$(grep "Number of files transferred:" $LOG_FILE)
num_t=${transferred#*: }
size=$(grep "Total transferred file size:" $LOG_FILE)
num_s=${size#*: }

# send notification
message="Transferred $num_t files ($num_s)"
terminal-notifier -title 'rsync Backup' -subtitle 'Backup complete' -message $message -execute "open $LOG_FILE" -sound Funk