# add required="true" to https://github.com/uzh-rpg/rpg_esim/blob/ada6795e6b33b2860089415d82e738544020523a/event_camera_simulator/esim_ros/launch/esim.launch
# see https://answers.ros.org/question/188052/close-roscore-after-running-with-roslaunch/
mkdir bag
for split in train val
do
    mkdir bag/$split
    if [ $split == 'train' ]
    then
        maxidx=239
    else
        maxidx=29
    fi
    for i in $(seq -f "%03g" 0 $maxidx)
    do
        echo `pwd`/bag/$split/$i.bag
        echo `pwd`/interpolated/$split/$i
        rosrun esim_ros esim_node \
         --data_source=2 \
         --path_to_output_bag=`pwd`/bag/$split/$i.bag \
         --path_to_data_folder=`pwd`/interpolated/$split/$i \
         --ros_publisher_frame_rate=120 \
         --exposure_time_ms=120.0 \
         --use_log_image=1 \
         --log_eps=0.01 \
         --contrast_threshold_pos=0.18 \
         --contrast_threshold_neg=0.18 \
         --contrast_threshold_sigma_pos=0.03 \
         --contrast_threshold_sigma_neg=0.03
    done
done

