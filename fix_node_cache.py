#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动修复 node.rs 中的缓存同步问题"""

def fix_node_rs():
    file_path = r"d:\serverless\serverless_sim\serverless_sim\src\node.rs"
    
    print("正在读取文件...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"文件共 {len(lines)} 行")
    
    # 修复1：第282行，将 false 改为 true，并更新注释
    if len(lines) > 282:
        line_282 = lines[281]  # 0-indexed
        if 'false // 容器不存在，不能驱逐' in line_282:
            print(f"✓ 找到第282行需要修复的代码")
            lines[281] = line_282.replace(
                'false // 容器不存在，不能驱逐',
                'true // 返回true让缓存策略移除这个无效条目'
            )
            # 也更新280行的日志
            if len(lines) > 280:
                line_280 = lines[279]
                if 'Container {} not found when checking eviction on node {}' in line_280:
                    lines[279] = line_280.replace(
                        'Container {} not found when checking eviction on node {}',
                        'Container {} not found when checking eviction on node {}, mark as evictable to clean up'
                    )
                    # 添加注释到279行
                    if 'None => {' in lines[278]:
                        lines[278] = lines[278].replace(
                            'None => {',
                            'None => {\n                            // 容器已不存在，标记为可驱逐（让缓存策略清理这个无效条目）'
                        )
        else:
            print(f"⚠ 第282行内容不匹配，可能已经修复过了")
            print(f"  当前内容: {line_282.strip()}")
    
    # 修复2：第293-297行，添加容器存在性检查
    # 寻找 "// 1. 将old unload掉" 这一行
    found_unload_section = False
    for i in range(290, min(300, len(lines))):
        if '// 1. 将old unload掉' in lines[i]:
            found_unload_section = True
            print(f"✓ 找到第{i+1}行需要修复的 unload 代码")
            
            # 检查是否已经修复
            if 'let old_fnid = old.unwrap()' in ''.join(lines[i:i+10]):
                print(f"⚠ unload 部分可能已经修复过了")
                break
            
            # 找到这个section的范围（到下一个注释或空行）
            # 替换整个 if block
            # 从当前行开始找到 "if old.is_some() {"
            if_start = i + 1
            while if_start < len(lines) and 'if old.is_some()' not in lines[if_start]:
                if_start += 1
            
            if if_start < len(lines):
                # 找到对应的结束 }
                indent_count = lines[if_start].index('if')
                if_end = if_start + 1
                brace_count = 1
                while if_end < len(lines) and brace_count > 0:
                    for char in lines[if_end]:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                break
                    if_end += 1
                
                # 生成新代码
                indent = ' ' * indent_count
                new_code = f"""{indent}// 1. 将old unload掉（但只在容器确实存在时）
{indent}if old.is_some() {{
{indent}    let old_fnid = old.unwrap();
{indent}    // 检查容器是否真的存在
{indent}    if self.container(old_fnid).is_some() {{
{indent}        self.try_unload_container(old_fnid, env, false);
{indent}        log::info!("节点{{}}移除容器{{}}", self.node_id, old_fnid);
{indent}    }} else {{
{indent}        log::info!("缓存策略返回的容器{{}}已不存在，只清理缓存记录", old_fnid);
{indent}    }}
{indent}}}
"""
                # 替换
                lines[i:if_end] = [new_code]
                print(f"✓ 已替换第{i+1}到第{if_end}行的代码")
            break
    
    if not found_unload_section:
        print("⚠ 未找到 unload 部分，可能已经修复或代码结构已改变")
    
    print("正在写入文件...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("\n" + "="*60)
    print("✅ 修复完成！")
    print("="*60)
    print("\n接下来请执行：")
    print("  1. 在 Terminal 1 停止旧服务（Ctrl+C）")
    print("  2. cd d:\\serverless\\serverless_sim\\serverless_sim")
    print("  3. cargo build --release")
    print("  4. cargo run --release")
    print("  5. 在 Terminal 2 运行：python batch_run.py")
    print()

if __name__ == '__main__':
    try:
        fix_node_rs()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请手动按照 URGENT_FIX_node_rs.md 中的说明进行修复")
