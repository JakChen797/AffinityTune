hid_dim_values=(128 128 64)
layers_values=(4 3 4)

for i in "${!hid_dim_values[@]}"; do
    hid_dim=${hid_dim_values[$i]}
    layers=${layers_values[$i]}

    echo "Training with hid_dim=$hid_dim, layers=$layers"
    
    # 运行global_test.py并传递参数
    python global_test.py --hid_dim $hid_dim --layers $layers
done
