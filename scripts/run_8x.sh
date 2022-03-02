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
        python run_8x.py \
            --in /media/author/file4T0/EventDeblur/data/REDS/resized/$split/$i \
            --out /media/author/file4T0/EventDeblur/data/REDS/interpolated/$split/$i
    done
done

