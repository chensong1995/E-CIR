for split in train val
do
    if [ $split == 'train' ]
    then
        file_count=16
    else
        file_count=2
    fi
    for file_idx in $(seq 0 $(($file_count-1)))
    do
        start_video_idx=$(($file_idx*15))
        end_video_idx=$(($start_video_idx+15))
        out_name=$split\_$file_idx.hdf5
        printf "%s\n%s\n%s\n%s\n%s\n%s\n" \
            "$(cat condor_template.txt)" \
            "arguments = create_hdf5.py --split $split --start_video_idx $start_video_idx --end_video_idx $end_video_idx --out_name $out_name" \
            "Log = condor_logs/$split$file_idx.log" \
            "Error = condor_logs/$split$file_idx.err" \
            "Output = condor_logs/$split$file_idx.out" \
            "Queue" > condor_$split\_$file_idx.desp
        condor_submit condor_$split\_$file_idx.desp
    done
done
