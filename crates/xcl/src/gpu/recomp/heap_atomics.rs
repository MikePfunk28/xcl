//! Whether the driver runs an atomic on a memory window taken from the descriptor heap
//! (`Options::heap_windows`) on that window only.
//!
//! The heap form reaches window k through `ResourceDescriptorHeap[NonUniformResourceIndex(..)]`,
//! so lanes of one wave can name different buffers. Intel Arc (Pro B70, driver 32.0.101.8976)
//! then applies a lane's atomic to the lane's buffer and to every buffer of a lower index that
//! the wave names; loads, stores and the switch over root UAVs are right there. With that, a
//! title's atomics would change memory of other windows, so the runtime keeps the switch.

use std::mem::transmute_copy;
use std::sync::atomic::{AtomicBool, Ordering};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

/// Lane l adds `1 << (8 * (l & 1))` to dword 0 of buffer `l & 1`, the way translated programs do
/// (with the old value taken): 32 lanes on each buffer.
const SRC: &str = r#"
#define RS "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), UAV(u0)"
RWByteAddressBuffer outb : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint l : SV_GroupThreadID) {
    if (l == 0u) {
        RWByteAddressBuffer b0 = ResourceDescriptorHeap[0u];
        RWByteAddressBuffer b1 = ResourceDescriptorHeap[1u];
        b0.Store(0u, 0u);
        b1.Store(0u, 0u);
    }
    AllMemoryBarrierWithGroupSync();
    uint k = l & 1u;
    RWByteAddressBuffer w = ResourceDescriptorHeap[NonUniformResourceIndex(k)];
    uint o;
    w.InterlockedAdd(0u, 1u << (8u * k), o);
    outb.Store(4u * l, o);
}
"#;

/// Dword 0 of the two buffers after `SRC`: each lane's add reached its own buffer and no other.
pub(crate) fn only_their_buffer(sums: [u32; 2]) -> bool {
    sums == [32, 32 << 8]
}

unsafe fn buffer(dev: &ID3D12Device, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES, uav: bool) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).map_err(|e| format!("buffer: {e}"))?;
    r.ok_or_else(|| "buffer".to_string())
}

/// Runs `SRC` on two buffers of a heap of its own: Ok(the adds stayed on their buffers).
/// Err: the check could not run (no DXC, no SM 6.6, a failed call).
pub unsafe fn probe(dev: &ID3D12Device) -> Result<bool, String> {
    let dxil = super::dxc::compile(SRC, "main", "cs_6_6", &[])?;
    let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).map_err(|e| format!("root signature: {e}"))?;
    let pso: ID3D12PipelineState = dev
        .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: transmute_copy(&rs),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
            ..Default::default()
        })
        .map_err(|e| format!("pipeline: {e}"))?;
    let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(|e| format!("queue: {e}"))?;
    let views: ID3D12DescriptorHeap = dev
        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 2, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 })
        .map_err(|e| format!("descriptor heap: {e}"))?;
    let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
    let bufs = [buffer(dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true)?, buffer(dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true)?];
    for (k, b) in bufs.iter().enumerate() {
        let mut at = views.GetCPUDescriptorHandleForHeapStart();
        at.ptr += k * inc;
        dev.CreateUnorderedAccessView(b, None::<&ID3D12Resource>, Some(&super::runner::raw_window_uav(256)), at);
    }
    let out = buffer(dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true)?;
    let readback = buffer(dev, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false)?;
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| format!("allocator: {e}"))?;
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).map_err(|e| format!("list: {e}"))?;
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| format!("fence: {e}"))?;
    list.SetDescriptorHeaps(&[Some(views.clone())]);
    list.SetComputeRootSignature(&rs);
    list.SetComputeRootUnorderedAccessView(0, out.GetGPUVirtualAddress());
    list.Dispatch(1, 1, 1);
    for (k, b) in bufs.iter().enumerate() {
        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: transmute_copy(b),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                }),
            },
        }]);
        list.CopyBufferRegion(&readback, 4 * k as u64, b, 0, 4);
    }
    let run = || -> Result<[u32; 2], String> {
        list.Close().map_err(|e| format!("Close: {e}"))?;
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(|e| e.to_string())?)]);
        queue.Signal(&fence, 1).map_err(|e| format!("Signal: {e}"))?;
        crate::gpu::fence::wait(&fence, 1, dev, None, 2000).map_err(|e| e.to_string())?;
        let mut m = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut m)).map_err(|e| format!("Map: {e}"))?;
        let got = std::slice::from_raw_parts(m as *const u32, 2);
        let got = [got[0], got[1]];
        readback.Unmap(0, None);
        Ok(got)
    };
    match run() {
        Ok(sums) => Ok(only_their_buffer(sums)),
        Err(e) => {
            // the dispatch may still be running: what it uses stays (`real::check_uma_import`)
            std::mem::forget((queue, alloc, list, fence, readback, views, bufs, out, pso, rs));
            Err(e)
        }
    }
}

static REFUSED: AtomicBool = AtomicBool::new(false);

/// Whether `usable` found the driver's atomics wrong: `pso::heap_windows` is off then.
pub fn refused() -> bool {
    REFUSED.load(Ordering::Relaxed)
}

/// The runtime's check before the windows go into the shared heap: false (and the switch form
/// from then on) when an atomic reaches a window it does not name. A check that could not run
/// leaves the heap form on, as before.
pub unsafe fn usable(dev: &ID3D12Device) -> bool {
    match probe(dev) {
        Ok(true) => {}
        Ok(false) => {
            REFUSED.store(true, Ordering::Relaxed);
            crate::xlog_warn!("memory windows: this driver applies an atomic on a window from the descriptor heap to other windows too – XCL_HEAP_WINDOWS=1 is not used, shaders switch over the windows");
        }
        Err(e) => crate::xlog_warn!("memory windows: the atomics of the descriptor-heap form were not checked ({e})"),
    }
    !refused()
}

#[cfg(test)]
mod tests {
    #[test]
    fn only_their_buffer_takes_32_adds_on_each() {
        assert!(super::only_their_buffer([32, 32 << 8]));
        // Intel Arc Pro B70: the adds of the lanes on buffer 1 arrive in buffer 0 too
        assert!(!super::only_their_buffer([0x2020, 0x2000]));
        assert!(!super::only_their_buffer([32, 0]));
    }
}
