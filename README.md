> [!WARNING] > **zybel** is still WIP. **Breaking changes** may occur

# Zinyn - ML and AI made possible for fun

## Goal

Create a framework for training DNNs in Zig (using hardware accelerators)
and compare results

## Why?

To quote a wise man: "For the joy of programming!"

## Features

- [x] Generic tensor using comptime
- [x] Binops such as add, sub, mul, div
- [x] General tensor ops (clamp, reshape, sum, min, max...)
- [x] SGD optimizer
- [x] Common loss functions such as mse and mae + 3 more
- [ ] Activation functions
- [ ] Simple layers
- [ ] Autograd
- [ ] Computational graph
- [ ] Hardware acceleration support

## Install (Per 0.14 beta)

Run this command in the parent directory of your project

```sh
zig fetch --save git+https://github.com/Jafagervik/zybel.git
```

Then add these lines to build.zig before b.installArtifact(exe)

```zig
const zybel = b.dependency("zybel", .{});

exe.root_module.addImport("zybel", zybel.module("zybel"));

```

## Example

```zig
const std = @import("std");

const zb = @import("zybel");
const Tensor = zb.Tensor;
const TF32 = Tensor(f32);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Creates a F32 Tensor of shape (1, 3, 3)
    var t: TF32 = try TF32.ones(allocator, &[_]u32{ 1, 3, 3 });
    defer t.deinit();

    std.debug.print("First value is {d:.2}\n", .{t.getFirst()});

    bon.setVal(0, 2.0);

    std.debug.print("First value is now {d:.2}\n", .{t.getFirst()});

    // Prints input about the tensor
    t.print();

    std.debug.print("Sum is {d:.2}\n", .{t.sum()});
}
```

### Contribute

Hell yea! Feel free to add suggestions/corrections. I am a mere mortal, not a god
