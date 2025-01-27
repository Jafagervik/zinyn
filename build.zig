const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // =================
    // Benchmarks
    // =================
    const bench_exe = b.addExecutable(.{
        .name = "zybelbench",
        .root_source_file = b.path("bench/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(bench_exe);

    const run_cmd = b.addRunArtifact(bench_exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("bench", "Benchmark the lib");
    run_step.dependOn(&run_cmd.step);

    // =================
    //  Zybel as module
    // =================

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
