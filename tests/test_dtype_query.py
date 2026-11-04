import torch, torchvulkan

if __name__ == "__main__":
	d = torch.device("vulkan", 0)
	props = torchvulkan.get_device_properties(d)
	for fn in ("has_fp64", "has_int64", "has_fp16", "has_bf16", "has_int16", "has_int8"):
		print(f"{fn}: {props.__getattribute__(fn)}")
	print("success!")
