img_folder="../test/generated"
neg_attrs=("Eyeglasses" "Bangs" "Bald" "Wearing_Hat")
model=7
for neg_attr in "${neg_attrs[@]}"; do
    echo "Extracting ${neg_attr}..."
    python sort_desired_undesired.py "${img_folder}" "../test/model_${model}_attr_sorted_${neg_attr}/" "models/model_${model}_epoch.pt" CelebA/Anno/attr_names.txt --neg_class "${neg_attr}"
done

chmod +x run_classifier_sort.sh
