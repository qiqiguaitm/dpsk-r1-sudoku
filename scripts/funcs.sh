get_newest_dir() {
    local dir=$1  # 要检查的目录

    # 检查目录是否存在
    if [ ! -d "$dir" ]; then
        echo "Error: Directory does not exist"
        return 1
    fi

    # 获取目录下所有子目录，并按修改时间排序（最新的排在最前）
    local dirs=($(ls -t -d "$dir"/*/ 2>/dev/null))

    # 检查子目录数量
    if [ ${#dirs[@]} -lt 1 ]; then
        echo "Error: Not enough directories"
        return 1
    fi

    # 取第二新的子目录，并获取其绝对路径
    local second_newest_dir=$(realpath "${dirs[0]}")

    # 输出结果
    echo "$second_newest_dir"
}

