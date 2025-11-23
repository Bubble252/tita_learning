#!/bin/bash
# 跑酷训练监控脚本

echo "========================================="
echo "TITA 跑酷训练监控"
echo "========================================="
echo ""

# 检查进程
if ps aux | grep -v grep | grep train_parkour > /dev/null; then
    echo "✅ 训练进程正在运行"
    echo ""
else
    echo "❌ 训练进程未运行"
    echo ""
fi

# 显示最新日志
if [ -f "parkour_training.log" ]; then
    echo "📊 最新训练日志："
    echo "-----------------------------------------"
    tail -n 30 parkour_training.log
    echo ""
    echo "-----------------------------------------"
    
    # 统计关键信息
    if grep -q "Learning iteration" parkour_training.log; then
        echo ""
        echo "📈 训练进度："
        grep "Learning iteration" parkour_training.log | tail -n 5
    fi
    
    if grep -q "Stage" parkour_training.log; then
        echo ""
        echo "📚 课程阶段："
        grep "Stage" parkour_training.log | tail -n 1
    fi
else
    echo "⚠️  日志文件不存在"
fi

echo ""
echo "========================================="
