#!/bin/bash

# this script updates 2d-discharge-nn on the laboratory NAS from antares
notify-send "created_models backup starting..."

mount -a
LOG_FILE="/home/jarl/2D-NN/backup_log.txt"
now=$(date)

cat << EOF > $LOG_FILE
**** rsync log file for created_models ****
Started on: $now

EOF

# upload files through rsync
rsync -avzP --stats --exclude '.git/' ~/2D-NN/created_models ~/jarl-nas/2d-discharge-nn/ >> $LOG_FILE
 
end=$(date)
printf "\nFinished on: $end" >> $LOG_FILE

# stats
transferred=$(grep "Number of files transferred:" $LOG_FILE)
num_t=${transferred#*: }
size=$(grep "Total transferred file size:" $LOG_FILE)
num_s=${size#*: }

# send notification
message="Transferred $num_t files ($num_s)"
notify-send "Backup complete."
terminal-notifier -group 'rsync-backup' -title 'rsync Backup' -subtitle 'Backup complete' -message $message -execute "open $LOG_FILE" -sound Funk
rsync -azP $LOG_FILE ~/jarl-nas/2d-discharge-nn/backup_log.txt