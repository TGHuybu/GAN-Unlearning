img_folder="CelebA/Img/img_align_celeba"
dst_folder="CelebA/Img/retained_positive"
attr_list="CelebA/Anno/attr_names.txt"          # List of attributes (no data)
attr_labels="CelebA/Anno/list_attr_celeba.txt"  # Labeled data table
neg_attrs=("Eyeglasses" "Bangs" "Bald" "Wearing_Hat")
for neg_attr in "${neg_attrs[@]}"; do
    echo "Removing ${neg_attr}..."
    python preprocess_data.py ${img_folder} ${dst_folder} ${attr_list} ${attr_labels} --neg_class ${neg_attr}
done

chmod +x preprocess.sh