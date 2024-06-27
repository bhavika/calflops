from calflops import calculate_flops
from torchvision import models

model = models.alexnet()
batch_size = 1

flops, macs, params = calculate_flops(
    model=model,
    input_shape=(batch_size, 3, 224, 224),
    output_as_string=False,
    print_results=True,
    print_detailed=True,
)

print(f"alexnet FLOPs:{flops}  MACs:{macs}   Params:{params}\n")


flops, macs, params = calculate_flops(
    model=model,
    input_shape=(batch_size, 3, 224, 224),
    print_results=False,
    print_detailed=False,
    output_as_string=True,
    output_precision=3,
    output_unit="M",
)

print(f"alexnet FLOPs:{flops}  MACs:{macs}   Params:{params}\n")
