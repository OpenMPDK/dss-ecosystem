#!/bin/bash
for i in {0..16000}
    do
	echo $i 
    	cp /opt/ansible/examples/data/mapillary/Africa/DaEwgFJdh-5zMGQTlRBqCg.jpg /mnt/pytorch-data/img_gen/$i.jpg
	convert -resize 90% /mnt/pytorch-data/img_gen/$i.jpg  /mnt/pytorch-data/img_1M/$i.jpg 
   done 
