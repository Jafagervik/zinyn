# Zinyn - ML and AI made possible for fun

To quote a wise man: "For the joy of programming!"

## Install

To add to build.zig.zon

```sh
zig fetch --save git+https://github.com/Jafagervik/zinyn/#Head
```

Then add to build.zig

```zig
const zinyn = b.dependency("zinyn", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("zinyn.zig", zinyn.module("zinyn.zig"));
```

## Example

```zig
const std = @import("std");

const zinyn = @import("zinyn");
const Tensor = zinyn.Tensor;
const TF32 = Tensor(f32);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var bon: TF32 = try TF32.ones(allocator, &[_]u32{ 1, 3, 3 });
    defer bon.deinit();

    std.debug.print("First value is {d:.3}\n", .{bon.getFirst()});

    bon.setVal(0, 2.0);

    std.debug.print("First value is {d:.3}\n", .{bon.getFirst()});

    std.debug.print("Shape is {any} and dtype is {any} \n", .{ bon.shapeIs(), bon.dtype() });

    std.debug.print("Sum is {d:.3}\n", .{bon.sum()});
}
```

### Contribute

Hell yea! Feel free to add suggestions/corrections. I am a mere mortal, not a god
