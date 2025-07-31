mkdir -p /people/cs/o/oxx220000/HiBug/hibug_simple

for dir in hibug_simple2 hibug_simple3; do
  src_dir="/people/cs/o/oxx220000/HiBug/$dir"
  for file in "$src_dir"/*.png; do
    base_name=$(basename "$file")
    dest="/people/cs/o/oxx220000/HiBug/hibug_simple/$base_name"
    
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