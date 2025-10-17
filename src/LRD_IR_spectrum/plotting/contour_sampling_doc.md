## 重构总结

### 模块化设计

原函数被拆分为 6 个独立的小函数:

1. **`extract_contour_paths`**: 提取等值线的所有路径顶点
2. **`calc_path_length` / `calc_all_path_lengths`**: 计算路径长度
3. **`merge_paths`**: 根据策略合并多条路径
4. **`sample_along_single_path`**: 在单条路径上均匀采样
5. **`sample_multiple_paths_separately`**: 对多条路径分别采样
6. **`sample_points_on_contour_v2`**: 高层组装函数

### 路径处理策略

支持 6 种路径处理策略（通过 `PathMergeStrategy` 枚举）:

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `LONGEST_ONLY` | 只取最长路径 | 等值线有多个小碎片,只关心主要部分 |
| `DIRECT_CONCAT` | 直接按顺序连接 | 路径顺序已知且合理 |
| `SORT_BY_START_X` | 按起点 x 坐标排序 | 希望从左到右遍历 |
| `SORT_BY_START_Y` | 按起点 y 坐标排序 | 希望从下到上遍历 |
| `SORT_BY_LENGTH` | 按长度降序排列 | 优先采样长路径 |
| `NEAREST_NEIGHBOR` | 最近邻贪心连接 | 希望路径尽量连续 |

### 独立采样模式

设置 `separate_sampling=True` 可对每条路径单独采样(不合并):
- `by_length_proportion=True`: 按长度比例分配采样点数
- `by_length_proportion=False`: 均匀分配采样点数

### 优势

1. **可读性**: 每个函数职责单一,易于理解和维护
2. **可测试性**: 每个小函数可独立测试
3. **可扩展性**: 容易添加新的合并策略或采样方法
4. **灵活性**: 可根据实际情况选择合适的路径处理方式
5. **向后兼容**: 默认行为(只取最长路径)与原函数基本一致

### 使用建议

- **简单场景**: 使用默认参数,与原函数行为一致
- **多个闭合环**: 根据需要选择合适的合并策略或使用独立采样
- **调试**: 先用 `inspect_contour_paths` 查看路径信息,再决定策略