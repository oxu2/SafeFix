mkdir -p /people/cs/o/oxx220000/safefix/safefix_simple

for dir in safefix_simple2 safefix_simple3; do
  src_dir="/people/cs/o/oxx220000/safefix/$dir"
  for file in "$src_dir"/*.png; do
    base_name=$(basename "$file")
    dest="/people/cs/o/oxx220000/safefix/safefix_simple/$base_name"
    
    if [ ! -e "$dest" ]; then
      mv "$file" "$dest"
    else
      i=1
      while [ -e "${dest%.*}_$i.png" ]; do
        ((i++))
      done
      mv "$file" "${dest%.*}_$i.png"
    fi
  done
done