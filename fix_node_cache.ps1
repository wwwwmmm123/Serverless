#!/usr/bin/env pwsh
# 自动修复 node.rs 中的缓存同步问题

$filePath = "d:\serverless\serverless_sim\serverless_sim\src\node.rs"

Write-Host "正在读取文件..." -ForegroundColor Yellow
$content = Get-Content $filePath -Raw -Encoding UTF8

Write-Host "正在应用修复..." -ForegroundColor Yellow

# 修复1：将驱逐检查中的 false 改为 true
$oldPattern1 = @"
                        None => \{
                            log::warn!\("Container \{\} not found when checking eviction on node \{\}", 
                                to_replace, node.node_id\);
                            false // 容器不存在，不能驱逐
                        \}
"@

$newPattern1 = @"
                        None => {
                            // 容器已不存在，标记为可驱逐（让缓存策略清理这个无效条目）
                            log::warn!("Container {} not found when checking eviction on node {}, mark as evictable to clean up", 
                                to_replace, node.node_id);
                            true // 返回true让缓存策略移除这个无效条目
                        }
"@

$content = $content -replace $oldPattern1, $newPattern1

# 修复2：更新 unload 逻辑
$oldPattern2 = @"
            // 1. 将old unload掉
            if old.is_some\(\) \{
                self.try_unload_container\(old.unwrap\(\), env, false\);
                log::info!\("节点\{\}移除容器\{\}", self.node_id, old.unwrap\(\)\);
            \}
"@

$newPattern2 = @"
            // 1. 将old unload掉（但只在容器确实存在时）
            if old.is_some() {
                let old_fnid = old.unwrap();
                // 检查容器是否真的存在
                if self.container(old_fnid).is_some() {
                    self.try_unload_container(old_fnid, env, false);
                    log::info!("节点{}移除容器{}", self.node_id, old_fnid);
                } else {
                    log::info!("缓存策略返回的容器{}已不存在，只清理缓存记录", old_fnid);
                }
            }
"@

$content = $content -replace $oldPattern2, $newPattern2

Write-Host "正在写入文件..." -ForegroundColor Yellow
$content | Set-Content $filePath -Encoding UTF8 -NoNewline

Write-Host "" 
Write-Host "✅ 修复完成！" -ForegroundColor Green
Write-Host ""
Write-Host "接下来请执行：" -ForegroundColor Cyan
Write-Host "  1. 在 Terminal 1 停止旧服务（Ctrl+C）" -ForegroundColor White
Write-Host "  2. cd d:\serverless\serverless_sim\serverless_sim" -ForegroundColor White
Write-Host "  3. cargo build --release" -ForegroundColor White
Write-Host "  4. cargo run --release" -ForegroundColor White
Write-Host "  5. 在 Terminal 2 运行：python batch_run.py" -ForegroundColor White
Write-Host ""
