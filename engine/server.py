"""
DAERWEN3 2D生态系统 - WebSocket实时服务器
将模拟数据实时推送到Web前端
"""
import asyncio
import json
import websockets
import time
import threading
import builtins
import sys
from typing import Optional, Union
from .core import Ecology2DSystem, Ecology2DConfig, ExternalInput, SystemOutput


def _safe_print(*args, **kwargs):
    """Best-effort console print that won't crash on narrow encodings (e.g. gbk)."""
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        fallback_args = []
        for arg in args:
            text = str(arg)
            safe_text = text.encode(encoding, errors="backslashreplace").decode(encoding, errors="ignore")
            fallback_args.append(safe_text)
        builtins.print(*fallback_args, **kwargs)


# Keep all module prints safe on Windows terminals with non-UTF8 encodings.
print = _safe_print  # type: ignore

# 优先使用orjson（快10倍），否则用标准json
try:
    import orjson
    def json_dumps(obj):
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY).decode('utf-8')
    print("✅ 使用orjson加速JSON序列化（10倍速）")
except ImportError:
    def json_dumps(obj):
        return json.dumps(obj, ensure_ascii=False)
    print("⚠️ 使用标准json（建议: pip install orjson）")


class Ecology2DServer:
    """2D生态系统实时服务器"""
    
    def __init__(
        self,
        config: Ecology2DConfig,
        external_control: bool = False,
        run_mode: str = 'demo',
        enable_autonomous_disturbance: Optional[bool] = None,
    ):
        self.config = config
        self.system = Ecology2DSystem(config)
        self.connected_clients = set()
        self.is_running = False
        self.is_paused = False
        self.update_interval = 0.0  # 不限速，CPU全速运行
        self.external_control = external_control
        valid_modes = {'baseline', 'experiment', 'demo'}
        if run_mode not in valid_modes:
            raise ValueError(f"run_mode 必须是 {sorted(valid_modes)} 之一，当前: {run_mode}")
        self.run_mode = run_mode
        if enable_autonomous_disturbance is None:
            # baseline 默认禁止服务层自主干预，experiment/demo 默认允许
            self.enable_autonomous_disturbance = run_mode != 'baseline'
        else:
            self.enable_autonomous_disturbance = bool(enable_autonomous_disturbance)
        self.manual_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # 动态扰动机制
        self.last_population = 0
        self.stagnation_counter = 0
        self.last_disturbance_step = 0
        self.disturbance_cooldown = 1200  # 扰动冷却时间（步数）
        
    async def websocket_handler(self, websocket):
        """处理WebSocket连接"""
        print(f"🔗 新客户端连接: {websocket.remote_address}")
        self.connected_clients.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    action = data.get('action')
                    
                    if action == 'pause':
                        self.is_paused = True
                        print("⏸️ 模拟已暂停")
                    elif action == 'resume':
                        self.is_paused = False
                        print("▶️ 模拟已继续")
                    elif action == 'reset':
                        self.reset_system()
                        print("🔄 系统已重置")
                    elif action == 'export':
                        self.export_data()
                        print("💾 数据已导出")
                    elif action == 'input':
                        input_type = data.get('input_type')
                        payload = data.get('payload', {})
                        metadata = data.get('metadata', {})
                        if input_type and isinstance(payload, dict):
                            event = ExternalInput(input_type=input_type, params=payload, metadata=metadata)
                            applied = self.system.apply_external_input(event)
                            latest = self.system.input_history[-1] if self.system.input_history else {}
                            ack = {
                                'type': 'input_ack',
                                'input': event.to_dict(),
                                'time_step': self.system.time_step,
                                'applied': bool(applied),
                                'status': latest.get('status', 'unknown'),
                                'reason': latest.get('reason', ''),
                            }
                            await websocket.send(json_dumps(ack))
                        else:
                            await websocket.send(json_dumps({
                                'type': 'error',
                                'message': '无效的输入参数'
                            }))
                    elif action == 'snapshot':
                        snapshot = self.system.get_system_output(metadata={'source': 'snapshot'})
                        await websocket.send(json_dumps({
                            'type': 'snapshot',
                            'data': snapshot.to_dict()
                        }))
                        
                except json.JSONDecodeError:
                    print(f"⚠️ 无法解析消息: {message}")
                except Exception as e:
                    print(f"❌ 消息处理错误: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"📱 客户端断开: {websocket.remote_address}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def broadcast_state(self):
        """广播系统状态到所有客户端"""
        if self.external_control:
            await self._broadcast_external()
            return
        while self.is_running:
            try:
                if not self.is_paused:
                    self.system.step()
                    system_output = self.system.get_system_output()
                    self._handle_stagnation(system_output)
                    message_json = json_dumps(system_output.to_dict())
                    if self.connected_clients:
                        websockets.broadcast(self.connected_clients, message_json)
                await asyncio.sleep(0)
            except Exception as e:
                print(f"❌ 广播错误: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)

    async def _broadcast_external(self):
        if self.manual_queue is None:
            self.manual_queue = asyncio.Queue()
        while self.is_running:
            try:
                snapshot = await self.manual_queue.get()
                if self.connected_clients:
                    message_json = json_dumps(snapshot)
                    websockets.broadcast(self.connected_clients, message_json)
            except Exception as e:
                print(f"❌ 外部广播错误: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(0.5)
    
    def reset_system(self):
        """重置系统"""
        self.system = Ecology2DSystem(self.config)
        print("✅ 系统已重置")

    def export_data(self):
        """导出当前数据"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ecology_2d_export_{timestamp}.json"
        
        snapshot = self.system.get_system_output(metadata={'source': 'export'})
        data = {
            'config': {
                'world_size': self.config.world_size,
                'n_particles': self.config.n_particles,
                'genome_length': self.config.genome_length,
            },
            'output': snapshot.to_dict(),
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 数据已导出到: {filename}")
    
    async def run(self, host='localhost', port=8765):
        """启动服务器"""
        self.is_running = True
        
        print(f"\n🌱 DAERWEN3 2D生态系统服务器")
        print(f"{'='*50}")
        print(f"🌍 世界大小: {self.config.world_size}×{self.config.world_size}")
        print(f"🧬 粒子数量: {self.config.n_particles}")
        print(f"🔬 基因长度: {self.config.genome_length}")
        print(f"🧭 运行模式: {self.run_mode}")
        print(f"⚙️ 自主扰动: {'开启' if self.enable_autonomous_disturbance else '关闭'}")
        print(f"🚀 服务地址: ws://{host}:{port}")
        print(f"{'='*50}\n")
        
        # 启动WebSocket服务器
        async with websockets.serve(self.websocket_handler, host, port):
            self._loop = asyncio.get_running_loop()
            if self.external_control:
                self.manual_queue = asyncio.Queue()
            # 启动广播任务
            broadcast_task = asyncio.create_task(self.broadcast_state())
            
            print(f"✅ 服务器已启动，等待连接...")
            print(f"💡 打开 ecology_2d/web.html 查看可视化\n")
            
            # 保持服务器运行
            await asyncio.Future()  # run forever

    def _handle_stagnation(self, system_output: SystemOutput):
        if not self.enable_autonomous_disturbance:
            return
        vis_data = system_output.visualization
        emergence_data = system_output.emergence
        current_pop = vis_data['stats']['alive_particles']
        time_step = system_output.time_step
        if abs(current_pop - self.last_population) < 5:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        if (self.stagnation_counter > 200 and
                time_step - self.last_disturbance_step > self.disturbance_cooldown and
                100 < current_pop < self.config.n_particles * 2):
            event_types = ['energy_fluctuation', 'mass_extinction', 'mutation_burst']
            event = event_types[time_step % len(event_types)]
            self.system.trigger_disturbance_event(event)
            self.last_disturbance_step = time_step
            self.stagnation_counter = 0
        self.last_population = current_pop
        if time_step % 100 == 0:
            print(
                f"步骤 {time_step}: "
                f"存活={current_pop} "
                f"能量={vis_data['stats']['total_energy']:.1f} "
                f"涌现={emergence_data['emergence_score']:.3f} "
                f"僵化={self.stagnation_counter}"
            )

    def push_external_snapshot(self, snapshot: Union[SystemOutput, dict]):
        """从外部测试推送系统状态到Web客户端"""
        if not self.external_control or self.manual_queue is None or self._loop is None:
            return
        if isinstance(snapshot, SystemOutput):
            payload = snapshot.to_dict()
        else:
            payload = snapshot
        try:
            asyncio.run_coroutine_threadsafe(self.manual_queue.put(payload), self._loop)
        except RuntimeError as exc:
            print(f"⚠️ 推送失败: {exc}")


def main():
    """主函数"""
    # 配置
    config = Ecology2DConfig(
        world_size=200,
        n_particles=3000,
        genome_length=32,
        mutation_rate=0.01,
        n_chemical_species=10,
    )
    
    # 创建服务器
    server = Ecology2DServer(config)
    
    # 启动
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n\n🛑 服务器已停止")


if __name__ == "__main__":
    main()
