# The following script download COCO-2017's train and val sets images then put them into a single fodler trainval2017/
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip && unzip val2017.zip
mv train2017 trainval2017 && mv val2017/* trainval2017 && rm -r val2017

# Copy only relevant images to images/
mkdir images
python select_only_relevant_images.py data_files/all_vsr_validated_data.jsonl/  trainval2017/ images/