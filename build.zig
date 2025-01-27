const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    _ = b.addModule("zybel", .{
        .root_source_file = b.path("src/zybel.zig"),
    });

    const tests = b.addTest(.{
        .target = target,
        .optimize = optimize,
        // .test_runner = b.path("test_runner.zig"),
        .root_source_file = b.path("src/zybel.zig"),
    });

    const run_lib_unit_tests = b.addRunArtifact(tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
}
