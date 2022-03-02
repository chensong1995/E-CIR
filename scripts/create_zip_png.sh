for split in train val
do
    if [ $split == 'train' ]
    then
        maxidx=239
    else
        maxidx=29
    fi
    for i in $(seq -f "%03g" 0 $maxidx)
    do
        python extract_events_from_rosbag.py `pwd`/bag/$split/$i.bag \
            --output_folder `pwd`/zip/$split \
            --output_corrupted_image_folder `pwd`/corrupted/$split \
            --event_topic /cam0/events \
            --corrupted_image_topic /cam0/image_corrupted
    done
done

