#!/usr/bin/env swift
// bench/bench.swift — repeatable llama-bench runs against pinned models,
// with the local llama-server out of the way and the results kept.
//
//   bench/bench.swift list [<selector>] [--fit vram|offload]
//   bench/bench.swift capabilities [<selector>]
//   bench/bench.swift fetch <selector> [--fit vram|offload] [--dry-run]
//   bench/bench.swift run <selector> <suite> [--build <dir> | --accelerator <name>] [--fit vram|offload]
//                                [--ncmoe <list>] [--threads <list>] [--kv f16|q8_0|q4_0] [--ub <n>] [--reps <n>]
//                                [--dry-run] [--store <sqlite>] [--ctx <n>] [--tasks <json>] [--source <dir>]
//                                [--repeat <n>] (reuse suite)
//
// A selector is a model name, a purpose (coding, diagramming, latex, prose,
// graphics, general), or `all`.
//
// Swift 6 or later: bench/setup-swift.sh checks for it and installs it.
//
// The binaries it runs come from one of two places:
//   - A build directory of this repository (--build <dir>; without --build and
//     without an install, <repo>/build/bin): llama-bench and llama-server run
//     from there directly, on whatever backend that build has (see
//     SQLITE-STORE.md, "Build").
//   - An install by the linux repository's 00-setup-llama-cpp.swift: its
//     llama-bench and llama-server through llama-on, on the accelerator
//     --accelerator names (default nvidia; with --vulkan also intel,
//     nvidia-vulkan and cpu). Used when /opt/llama.cpp/current/bin/llama-on
//     exists and --build is not given.
//
// Runs entirely as you, and never as root: it calls no sudo and refuses to run
// under it. Root's work (the services, the system model store) belongs to
// 00-setup-llama-cpp.swift; this script only measures, and needs no root for it.
//
// Why it is shaped this way:
//   - Models are declared here, pinned by Hugging Face revision and SHA-256,
//     so a run names exactly what it measured.
//   - `fetch` downloads as you into your own store, ~/.local/share/llama-models
//     ($XDG_DATA_HOME): beside the destination, SHA-256 checked, then moved
//     into place. The service's store, /var/lib/llama-models (root-owned), is
//     read where a model is already there, and never written.
//   - Each model declares its purposes, primary first, and is selected by
//     them, in this order:
//       coding       writing, editing and debugging code; agent tool use
//       diagramming  diagrams as code: Mermaid, PlantUML, Graphviz, TikZ, SVG
//       latex        LaTeX documents and mathematics
//       prose        writing and editing
//       graphics     reading images: screenshots, figures, diagrams, scanned
//                    formulas (needs the model's image projector, fetched with it).
//                    Generating images is a different kind of model (diffusion)
//                    and out of scope here.
//       general      everything else
//     Placement is the second criterion: `vram` models fit the 24 GB dGPU
//     whole; `offload` models keep expert weights in system RAM (--n-cpu-moe).
//     Placement is the dGPU's: the iGPU's memory is system RAM (a 92 GiB
//     Vulkan heap), so every declared model fits it whole.
//   - --accelerator selects the device. Off the dGPU (intel, cpu) nothing here
//     calls nvidia-smi, which would wake the dGPU: no dGPU facts, no GPU
//     sampling, no expert offload unless --ncmoe asks for it. On cpu, without
//     --threads, baseline and depth sweep 8, 16 and 24 threads: llama-bench's
//     own default is 4 here.
//   - llama-bench measures text only. A graphics model's projector is fetched
//     so the model is usable, but the image encoder's speed is not benchmarked.
//   - llama-bench loads its own copy of the model, so nothing else may hold
//     the device. `run` refuses while any llama-server@<accelerator>.service is
//     running (each stops by itself after its idle time) or, on the dGPU,
//     while any compute process holds it. The services' sockets keep listening,
//     since stopping them needs root: a client connecting mid-run starts a
//     server, which `run` detects from the server's invocation ID, and that
//     model's report and exit status say the run is invalid.
//   - Suites are declared, so the same suite on the same model is the same
//     command every time:
//       baseline  pp512 and tg128 at empty context
//       depth     the same at 0, 8k, 32k and 64k tokens of context
//       offload   a sweep of --n-cpu-moe x threads, for models too large for
//                 VRAM (expert weights in system RAM, the rest on the GPU)
//   - The KV cache is f16 unless --kv says otherwise. A dense model's cache
//     can outgrow the GPU at long context (Gemma 4 31B at 64k on 24 GB);
//     --kv q8_0 halves it, at a small quality cost, and the report records it.
//   - The reuse suite runs real tasks, not llama-bench: it starts its own
//     llama-server (as you, on 127.0.0.1:9940, at --ctx), and the model answers
//     a task set (reuse-tasks-c++.json beside this script: the sound-design C++
//     profile) through file tools. With --store, it also gets store tools over
//     an SQLite file you own: entries are append-only, keyed by the SHA-256 of
//     the files they derive from, and valid only while those are unchanged.
//     Per task it records outcome, correctness, peak context, prompt tokens
//     sent and computed, prompt and generation time, tool calls, store hits and
//     puts. Run once without --store (baseline), then twice with the same
//     --store (cold, then warm).
//   - Results go to ~/.local/share/llama-bench/<UTC stamp>-<model>-<suite>.{jsonl,md}:
//     llama-bench's own JSONL, and a Markdown report with the host, driver,
//     memory, build and model pin it ran against.
//   - Off nvidia, the file names carry the accelerator: <base>-<accelerator>.
//   - While each model runs on the dGPU, nvidia-smi samples it every 0.5 s into
//     <base>-gpu.csv (power, SM and memory clocks, temperature, utilization,
//     VRAM, and the clock-limit reasons). The GPU is awake for the run anyway;
//     the sampler starts after the run's checks and stops with it. It is a
//     thread in this process running one query per tick, never nvidia-smi's
//     own loop, so a killed benchmark cannot leave a poller holding the dGPU
//     awake. The report
//     gains a "GPU during the run" section over the samples under load, so a
//     slow result shows whether the GPU was power- or thermally limited.
//
// Disciplines:
//   - argv only; no shell is ever invoked.
//   - Queries always run; mutations print, and run unless --dry-run.
//   - One authority: the invoking user's.

import Foundation

#if canImport(Glibc)
    import Glibc
#endif

setbuf(stdout, nil)

// MARK: - Process

func fatal(_ lines: String...) -> Never {
    for line in lines { print("Fatal: \(line)") }
    exit(1)
}

struct Captured {
    let status: Int32
    let output: String
    var succeeded: Bool { status == 0 }
}

func query(_ argv: [String]) -> Captured {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = argv
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = FileHandle.nullDevice
    do { try process.run() } catch { return Captured(status: 127, output: "") }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return Captured(status: process.terminationStatus, output: String(decoding: data, as: UTF8.self))
}

func trimmed(_ text: String) -> String { text.trimmingCharacters(in: .whitespacesAndNewlines) }

// Runs with the terminal attached; stdout optionally to a file.
func spawn(_ argv: [String], stdoutTo file: String? = nil) -> Int32 {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = argv
    if let file {
        _ = FileManager.default.createFile(atPath: file, contents: nil)
        guard let handle = FileHandle(forWritingAtPath: file) else { fatal("cannot write \(file)") }
        process.standardOutput = handle
    }
    do { try process.run() } catch { return 127 }
    process.waitUntilExit()
    return process.terminationReason == .uncaughtSignal ? 128 + process.terminationStatus : process.terminationStatus
}

var dryRun = false

func mutate(_ argv: [String]) {
    print("    >>> \(argv.joined(separator: " "))")
    guard !dryRun else { return }
    let status = spawn(argv)
    if status != 0 { fatal("command failed (exit \(status)): \(argv.joined(separator: " "))") }
}

// MARK: - Declarations

let dataHome = ProcessInfo.processInfo.environment["XDG_DATA_HOME"]
    ?? "\(FileManager.default.homeDirectoryForCurrentUser.path)/.local/share"
// Yours: fetch writes here.
let userModelDirectory = "\(dataHome)/llama-models"
// The service's (00-setup-llama-cpp.swift, root-owned): read, never written.
let systemModelDirectory = "/var/lib/llama-models"
let buildInfo = "/opt/llama.cpp/current/BUILD-INFO"
let resultsDirectory = "\(dataHome)/llama-bench"

// The accelerators of 00-setup-llama-cpp.swift, read from their files as
// llama-on reads them: llama-on <accelerator> <command> runs the command from
// the accelerator's build with only its device visible.
let llamaOn = "/opt/llama.cpp/current/bin/llama-on"
let acceleratorDirectory = "/etc/llama.cpp/accelerators"
let cudaBin = "/opt/llama.cpp/current/bin"

struct Accelerator {
    let name: String
    let summary: String
    // Runs on the dGPU: the CUDA build, or Vulkan pointed at NVIDIA's driver.
    // Otherwise nothing here may call nvidia-smi, which would wake the dGPU.
    let usesDGPU: Bool
    // The CPU alone (the Vulkan build with Vulkan disabled).
    let cpuOnly: Bool
}

// The thread counts a CPU-only run sweeps when --threads is not given:
// llama-bench's own default here is 4 threads, a sixth of this CPU (measured
// 2026-11-13). 8 is the P-cores; 16 adds half the E-cores; 24 is every core.
let cpuThreadsSweep = [8, 16, 24]

func installedAccelerator(_ name: String) -> Accelerator? {
    guard let text = try? String(contentsOfFile: "\(acceleratorDirectory)/\(name).env", encoding: .utf8) else { return nil }
    let lines = text.split(separator: "\n").map(String.init)
    func value(_ key: String) -> String? {
        lines.first { $0.hasPrefix("\(key)=") }.map { String($0.dropFirst(key.count + 1)) }
    }
    let summary = lines.first.map { $0.hasPrefix("# ") ? String($0.dropFirst(2)) : $0 } ?? name
    let usesDGPU = value("LLAMA_ON_BIN") == cudaBin || (value("VK_ICD_FILENAMES")?.contains("nvidia") ?? false)
    return Accelerator(name: name, summary: summary, usesDGPU: usesDGPU, cpuOnly: value("GGML_DISABLE_VULKAN") == "1")
}

func installedAcceleratorNames() -> [String] {
    ((try? FileManager.default.contentsOfDirectory(atPath: acceleratorDirectory)) ?? [])
        .filter { $0.hasSuffix(".env") }.map { String($0.dropLast(4)) }.sorted()
}

// The micro-batch (llama.cpp -ub), when not llama.cpp's default of 512. On
// the iGPU it bounds how long one compute submission runs: i915 resets a
// context whose submission does not yield within the engine's preempt
// timeout (7.5 s on rcs0/ccs0), and those engines also draw the desktop.
var ubatch: Int?

var acceleratorName = "nvidia"
var acceleratorGiven = false
// Resolved once the arguments are read.
var accelerator = Accelerator(name: "nvidia", summary: "", usesDGPU: true, cpuOnly: false)

// Build mode: a build directory of this repository, its binaries run directly
// (--build, or <repo>/build/bin when nothing is installed). nil: the install,
// through llama-on.
var buildDirectory: String?
let repositoryRoot = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
    .deletingLastPathComponent().deletingLastPathComponent().path
let defaultBuildDirectory = "\(repositoryRoot)/build/bin"

// A build directory's backend, from what its llama-bench links: the CUDA
// driver library means the NVIDIA GPU. Reading the links runs nothing on a GPU.
func buildAccelerator(_ directory: String) -> Accelerator {
    let bench = "\(directory)/llama-bench"
    guard FileManager.default.isExecutableFile(atPath: bench) else {
        fatal("no llama-bench in \(directory); build it first (SQLITE-STORE.md, \"Build\"), or pass --build <dir>.")
    }
    let links = query(["ldd", bench]).output
    let cuda = links.contains("libcuda.so")
    let vulkan = links.contains("libvulkan.so")
    let backend = cuda ? "CPU + CUDA" : (vulkan ? "CPU + Vulkan" : "CPU")
    return Accelerator(name: "build", summary: "\(backend), \(directory)", usesDGPU: cuda, cpuOnly: !cuda && !vulkan)
}

// A llama.cpp command on the selected accelerator.
func onAccelerator(_ command: String) -> [String] {
    if let directory = buildDirectory { return ["\(directory)/\(command)"] }
    return [llamaOn, accelerator.name, command]
}

// The llama.cpp version of the binaries in use, for every report.
func buildVersion() -> String {
    if let directory = buildDirectory {
        // --version prints to stderr; query keeps stdout only
        return query(["sh", "-c", "exec \"$0\" --version 2>&1", "\(directory)/llama-bench"]).output.split(separator: "\n")
            .filter { $0.hasPrefix("version:") }.joined(separator: " ")
    }
    return ((try? String(contentsOfFile: buildInfo, encoding: .utf8)) ?? "").split(separator: "\n").filter { $0.hasPrefix("tag=") || $0.hasPrefix("commit=") }
        .joined(separator: " ")
}

// The services of 00-setup-llama-cpp.swift (llama-server@<accelerator>.service),
// one per accelerator that serves. Their sockets may keep listening: nothing
// here needs root. A server already running would hold its device's memory, so
// run refuses; one started during a run (a client connected) makes the run
// invalid, which run detects from the server's invocation ID and reports.
func serverUnits() -> [String] {
    query(["systemctl", "list-units", "--all", "--plain", "--no-legend", "--type=service", "llama-server@*"]).output
        .split(separator: "\n").compactMap { $0.split(separator: " ").first.map(String.init) }
        .filter { $0.hasPrefix("llama-server@") }
}

func invocationIDs(_ units: [String]) -> [String: String] {
    Dictionary(uniqueKeysWithValues: units.map {
        ($0, trimmed(query(["systemctl", "show", "--property=InvocationID", "--value", $0]).output))
    })
}

// A file of the right size in either store; your store first.
func locate(_ name: String, _ bytes: Int64) -> String? {
    [userModelDirectory, systemModelDirectory].map { "\($0)/\(name)" }.first { fileSize($0) == bytes }
}

struct Offload {
    let moeLayers: Int        // layers whose experts --n-cpu-moe can move to RAM
    let ncmoeSweep: [Int]     // the offload suite's default sweep
    let threadsSweep: [Int]
    let safeNcmoe: Int        // used by baseline and depth when --ncmoe is not given
}

// The order is the selection order: coding models first.
enum Purpose: String, CaseIterable, Comparable {
    case coding, diagramming, latex, prose, graphics, general

    static func < (lhs: Purpose, rhs: Purpose) -> Bool {
        allCases.firstIndex(of: lhs)! < allCases.firstIndex(of: rhs)!
    }
}

enum Fit: String { case vram, offload }

// A pinned companion file, e.g. the image projector (--mmproj) of a vision model.
struct Companion {
    let file: String
    let sha256: String
    let bytes: Int64
}

struct Model {
    let name: String
    let purposes: [Purpose]   // primary first
    let note: String
    let repository: String
    let revision: String
    let file: String
    let sha256: String
    let bytes: Int64
    let offload: Offload?     // nil: fits in VRAM whole
    var projector: Companion? = nil
    // What the model takes and does, as its files declare it; checked against
    // them by `capabilities` and after `fetch`. nil: not read yet (not fetched).
    var capabilities: Capabilities? = nil

    // Each companion is stored beside the model under a name carrying the
    // model's, since every repository calls its projector mmproj-*.gguf.
    var projectorPath: String? {
        projector.map { locate("\(name).\($0.file)", $0.bytes) ?? "\(userModelDirectory)/\(name).\($0.file)" }
    }
    var projectorURL: String? { projector.map { "https://huggingface.co/\(repository)/resolve/\(revision)/\($0.file)" } }
    var present: Bool {
        fileSize(path) == bytes && (projector == nil || fileSize(projectorPath!) == projector!.bytes)
    }
    var downloadBytes: Int64 { bytes + (projector?.bytes ?? 0) }

    var url: String { "https://huggingface.co/\(repository)/resolve/\(revision)/\(file)" }
    // Where it is, or where fetch puts it.
    var path: String { locate(file, bytes) ?? "\(userModelDirectory)/\(file)" }
    var primary: Purpose { purposes[0] }
    var fit: Fit { offload == nil ? .vram : .offload }
}

// MARK: - Capabilities

// What a model accepts. An image needs the model's projector, whose
// metadata declares its vision encoder.
enum Input: String, CaseIterable { case text, image }

// The chat-template switch that turns a model's thinking down or off.
enum ThinkingSwitch: String, CaseIterable {
    case reasoningEffort = "reasoning_effort"
    case enableThinking = "enable_thinking"
}

// Every field is read from the model's own files: the GGUF metadata of the
// model and of its projector, and the chat template the model carries.
struct Capabilities: Equatable {
    let inputs: [Input]
    let toolCalls: Bool              // the chat template renders tool calls
    let thinking: [ThinkingSwitch]   // the switches the chat template reads
    let contextLength: Int           // <architecture>.context_length: the native maximum
    let knowledgeCutoff: String?     // as the chat template states it; nil: it states none

    var row: [String] {
        [inputs.map(\.rawValue).joined(separator: ", "),
         toolCalls ? "yes" : "no",
         thinking.isEmpty ? "none" : thinking.map { "`\($0.rawValue)`" }.joined(separator: ", "),
         contextLength.formatted(),
         knowledgeCutoff ?? "not stated"]
    }
}

// MARK: GGUF metadata

// A GGUF file starts with its metadata: the magic "GGUF", a version, the tensor
// count, the key-value count, then each key (a length-prefixed string), its
// value type and its value, little-endian. Only scalars and strings are kept;
// arrays (the tokenizer's vocabulary) are read past.
enum GGUFValue: Equatable {
    case integer(Int64)
    case float(Double)
    case bool(Bool)
    case string(String)
    case array
}

struct GGUFError: Error, CustomStringConvertible {
    let description: String
}

final class GGUFReader {
    private let handle: FileHandle
    private var buffer = Data()
    private var position = 0

    init(_ path: String) throws {
        guard let handle = FileHandle(forReadingAtPath: path) else { throw GGUFError(description: "cannot open \(path)") }
        self.handle = handle
    }

    deinit { try? handle.close() }

    private func need(_ count: Int) throws {
        while buffer.count - position < count {
            let chunk = handle.readData(ofLength: 1 << 20)
            if chunk.isEmpty { throw GGUFError(description: "the metadata ends early") }
            buffer = buffer.subdata(in: position..<buffer.count) + chunk
            position = 0
        }
    }

    func bytes(_ count: Int) throws -> Data {
        try need(count)
        defer { position += count }
        return buffer.subdata(in: position..<position + count)
    }

    func skip(_ count: Int) throws {
        try need(count)
        position += count
    }

    private func integer<T: FixedWidthInteger>(_: T.Type) throws -> T {
        T(littleEndian: try bytes(MemoryLayout<T>.size).withUnsafeBytes { $0.loadUnaligned(as: T.self) })
    }

    func u32() throws -> UInt32 { try integer(UInt32.self) }
    func u64() throws -> UInt64 { try integer(UInt64.self) }

    func string() throws -> String {
        String(decoding: try bytes(Int(try u64())), as: UTF8.self)
    }

    // GGUF value types 0–12.
    func value(_ type: UInt32) throws -> GGUFValue {
        switch type {
        case 0: return .integer(Int64(try integer(UInt8.self)))
        case 1: return .integer(Int64(try integer(Int8.self)))
        case 2: return .integer(Int64(try integer(UInt16.self)))
        case 3: return .integer(Int64(try integer(Int16.self)))
        case 4: return .integer(Int64(try integer(UInt32.self)))
        case 5: return .integer(Int64(try integer(Int32.self)))
        case 6: return .float(Double(Float(bitPattern: try integer(UInt32.self))))
        case 7: return .bool(try integer(UInt8.self) != 0)
        case 8: return .string(try string())
        case 9:
            let elementType = try u32()
            let count = try u64()
            let width: [UInt32: Int] = [0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8]
            if let size = width[elementType] {
                try skip(Int(count) * size)
            } else {
                for _ in 0..<count { _ = try value(elementType) }
            }
            return .array
        case 10: return .integer(Int64(bitPattern: try u64()))
        case 11: return .integer(try integer(Int64.self))
        case 12: return .float(Double(bitPattern: try u64()))
        default: throw GGUFError(description: "unknown value type \(type)")
        }
    }
}

func ggufMetadata(_ path: String) throws -> [String: GGUFValue] {
    let reader = try GGUFReader(path)
    guard try reader.bytes(4) == Data("GGUF".utf8) else { throw GGUFError(description: "\(path) is not a GGUF file") }
    let version = try reader.u32()
    guard version >= 2 else { throw GGUFError(description: "\(path): GGUF version \(version) is not supported") }
    _ = try reader.u64()  // tensors
    var metadata: [String: GGUFValue] = [:]
    for _ in 0..<(try reader.u64()) {
        let key = try reader.string()
        metadata[key] = try reader.value(try reader.u32())
    }
    return metadata
}

// Reads a fetched model's capabilities from its files.
func readCapabilities(_ model: Model) throws -> Capabilities {
    let metadata = try ggufMetadata(model.path)
    guard case .string(let architecture)? = metadata["general.architecture"] else {
        throw GGUFError(description: "\(model.file): no general.architecture")
    }
    guard case .integer(let context)? = metadata["\(architecture).context_length"] else {
        throw GGUFError(description: "\(model.file): no \(architecture).context_length")
    }
    guard case .string(let template)? = metadata["tokenizer.chat_template"] else {
        throw GGUFError(description: "\(model.file): no chat template")
    }
    var inputs: [Input] = [.text]
    if let path = model.projectorPath, let projector = model.projector, fileSize(path) == projector.bytes,
        case .bool(true)? = try ggufMetadata(path)["clip.has_vision_encoder"] {
        inputs.append(.image)
    }
    let cutoff = template.range(of: "Knowledge cutoff: ").map { range in
        String(template[range.upperBound...].prefix { $0.isNumber || $0 == "-" })
    }
    return Capabilities(inputs: inputs, toolCalls: template.contains("tool_calls"),
        thinking: ThinkingSwitch.allCases.filter { template.contains($0.rawValue) },
        contextLength: Int(context), knowledgeCutoff: cutoff)
}

// Compares each fetched model's declared capabilities with its files. Returns
// whether every one matched; prints the table MODELS.md shows.
func checkCapabilities(_ selected: [Model]) -> Bool {
    var matched = true
    var rows: [String] = []
    for model in selected {
        guard let declared = model.capabilities else {
            guard model.present else {
                print("    \(model.name): not fetched, none declared")
                rows.append("| \(model.name) | *not fetched* | | | | |")
                continue
            }
            // Fetched but undeclared: show what its files say, to declare.
            matched = false
            do {
                let read = try readCapabilities(model)
                print("    [FAIL] \(model.name): none declared; its files say " + read.row.joined(separator: "; "))
                rows.append("| \(model.name) | " + read.row.joined(separator: " | ") + " |")
            } catch {
                print("    [FAIL] \(model.name): none declared, and its files cannot be read: \(error)")
            }
            continue
        }
        guard model.present else {
            print("    \(model.name): not fetched; the declaration is not checked")
            rows.append("| \(model.name) | " + declared.row.joined(separator: " | ") + " |")
            continue
        }
        do {
            let read = try readCapabilities(model)
            if read == declared {
                print("    [ OK ] \(model.name): " + read.row.joined(separator: "; "))
            } else {
                matched = false
                print("    [FAIL] \(model.name): declared " + declared.row.joined(separator: "; "))
                print("           read from its files " + read.row.joined(separator: "; "))
            }
            rows.append("| \(model.name) | " + read.row.joined(separator: " | ") + " |")
        } catch {
            matched = false
            print("    [FAIL] \(model.name): \(error)")
        }
    }
    print("\n| Model | Input | Tool calls | Thinking switch | Native context | Knowledge cutoff (stated by the model file) |")
    print("|---|---|---|---|---:|---|")
    for row in rows { print(row) }
    return matched
}

// Chosen 2026-11-12 from the publishers' Hugging Face organizations and their
// model cards (vendor-reported benchmarks; few independent results exist yet),
// limited to architectures this llama.cpp release supports and to what this
// machine runs: 24 GB of VRAM whole, or 122 GB of RAM with expert offload.
// Superseded and dropped: Qwen3-Coder-30B, Qwen3-30B-2507, Qwen3-VL-30B
// (by Qwen 3.6/3.8, which see images natively), Devstral Small 2507 (by
// Devstral Small 2), Gemma 3 27B and Mistral Small 3.2 (by Gemma 4).
// Not added: Qwen3.8-Flash-Next (180B), GLM-5.3-Flash (321B), DeepSeek-V4.1-Flash
// (763B), too large even offloaded; GLM-4.7-Flash and Mistral Small 4 (119B), no
// independent results yet.
//
// Quantization: Q4_K_M, or the publisher's quantization-aware Q4_0 where one
// exists (Gemma 4), or the native MXFP4 (gpt-oss). "vram" means the weights fit
// the 24 GB GPU whole; the desktop is on the iGPU, so the dGPU's memory is the model's.
let declaredModels: [Model] = [
    // Coding.
    Model(name: "qwen3.8-27b", purposes: [.coding, .diagramming, .latex, .graphics, .general],
        note: "Qwen3.8 27B: dense, thinking (reasoning_effort), vision built in, 262k context; "
            + "vendor: SWE-bench Pro 61.7, LiveCodeBench v6 90.3, Terminal Bench 2.1 73.0",
        repository: "unsloth/Qwen3.8-27B-GGUF",
        revision: "4af51b3542f775b0a8e4b310bfb3f17e01c306c9", file: "Qwen3.8-27B-UD-Q4_K_M.gguf",
        sha256: "1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529", bytes: 16_564_259_744,
        offload: nil,
        projector: Companion(file: "mmproj-F16.gguf",
            sha256: "cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e", bytes: 927_607_488),
        capabilities: Capabilities(inputs: [.text, .image], toolCalls: true,
            thinking: [.reasoningEffort, .enableThinking], contextLength: 262_144, knowledgeCutoff: nil)),
    Model(name: "qwen3.6-35b-a3b", purposes: [.coding, .graphics, .diagramming, .general],
        note: "Qwen3.6 35B-A3B: MoE, 3B active, vision built in; the fast coder (tight fit: 20.5 GB)",
        repository: "ggml-org/Qwen3.6-35B-A3B-GGUF",
        revision: "52f9f9b8052fb9c8b69bba4a95eda0f319ac3d2a", file: "Qwen3.6-35B-A3B-Q4_K_M.gguf",
        sha256: "f4ffe03c9139e6a905a5d346f77d9fabdd179367c3280dfd005d1e75ca5ed8c8", bytes: 20_542_732_576,
        offload: nil,
        projector: Companion(file: "mmproj-Qwen3.6-35B-A3B-BF16.gguf",
            sha256: "360c746ddf38fb67c06f83a0a88742492aa231e59e335cd980e6302f59ea6089", bytes: 902_822_144),
        capabilities: Capabilities(inputs: [.text, .image], toolCalls: true,
            thinking: [.enableThinking], contextLength: 262_144, knowledgeCutoff: nil)),
    // The same model and revision in Q8_0: compares quantization kernels on one
    // model (on the iGPU, Q8_0 against Q4_K_M). 36.9 GB: on the dGPU, expert
    // weights in RAM; its 40 layers each hold 256 experts, 8 used per token.
    Model(name: "qwen3.6-35b-a3b-q8", purposes: [.coding, .general],
        note: "Qwen3.6 35B-A3B in Q8_0: the Q4_K_M model's weights at 8 bits, to compare quantizations",
        repository: "ggml-org/Qwen3.6-35B-A3B-GGUF",
        revision: "52f9f9b8052fb9c8b69bba4a95eda0f319ac3d2a", file: "Qwen3.6-35B-A3B-Q8_0.gguf",
        sha256: "f189cedbe18dc524dd74014b23431081262bc9fcc57c18fec7a7a25149246570", bytes: 36_903_139_616,
        offload: Offload(moeLayers: 40, ncmoeSweep: [40, 32, 24], threadsSweep: [8, 16], safeNcmoe: 40),
        // No projector declared for this file, so text only here.
        capabilities: Capabilities(inputs: [.text], toolCalls: true,
            thinking: [.enableThinking], contextLength: 262_144, knowledgeCutoff: nil)),
    Model(name: "devstral-small-2", purposes: [.coding],
        note: "Devstral Small 2 24B (2512): dense; software-engineering agents",
        repository: "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",
        revision: "6e458b8add42681bfd023de5eab93637694aaf82", file: "Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf",
        sha256: "d14ba9edee1bb4c4996a726deb81e49ae81800a3216f0774634238c380aee496", bytes: 14_334_446_752,
        offload: nil),
    // Prose.
    Model(name: "gemma-4-31b", purposes: [.prose, .graphics, .latex, .general],
        note: "Gemma 4 31B: dense, thinking, vision, 256k context, Google's QAT Q4_0; "
            + "vendor: AIME 2026 89.2, GPQA 84.3, MRCR 128k 66.4",
        repository: "google/gemma-4-31B-it-qat-q4_0-gguf",
        revision: "59dde24573e7e61570dba08b18a2e1fe246955ed", file: "gemma-4-31B_q4_0-it.gguf",
        sha256: "179cfb99212709597eae5929112cfca677e1bbf566178b479ae1da0c4772874b", bytes: 17_651_001_568,
        offload: nil,
        projector: Companion(file: "gemma-4-31B-it-mmproj.gguf",
            sha256: "6bd60bdb958548b4093196d38744b0f2290c12503a3fddd7486bffa9c5eb07a4", bytes: 1_200_726_368),
        capabilities: Capabilities(inputs: [.text, .image], toolCalls: true,
            thinking: [.enableThinking], contextLength: 262_144, knowledgeCutoff: nil)),
    Model(name: "gemma-4-26b-a4b", purposes: [.prose, .graphics, .general],
        note: "Gemma 4 26B-A4B: MoE, 4B active, vision, Google's QAT Q4_0; the fast writer",
        repository: "google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
        revision: "d1c082be9cf3c8a514acf63b8761f4b41935842e", file: "gemma-4-26B_q4_0-it.gguf",
        sha256: "3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d", bytes: 14_439_363_584,
        offload: nil,
        projector: Companion(file: "gemma-4-26B-it-mmproj.gguf",
            sha256: "a359953a076b877db30c31dbbb4c6d93b4a6e017ee5db5784247e4d4c0dd4f3b", bytes: 1_194_828_160),
        capabilities: Capabilities(inputs: [.text, .image], toolCalls: true,
            thinking: [.enableThinking], contextLength: 262_144, knowledgeCutoff: nil)),
    // General purpose.
    // 63 GB: does not fit the 24 GB GPU. Expert weights of the first N of
    // its 36 layers stay in RAM (--n-cpu-moe N); lower N is faster until VRAM runs out.
    // The 275HX has 8 P-cores and 16 E-cores, hence the 8 vs 16 thread sweep.
    // Measured 2026-11-14 (igpu-workloads.md): 24 at 8 threads is the best
    // (21.9 t/s); 23 is the same; 22 does not load. baseline and depth use 24.
    Model(name: "gpt-oss-120b", purposes: [.general, .latex, .coding, .diagramming],
        note: "gpt-oss 120B: MoE, 5.1B active; the largest model this machine runs (offloaded)",
        repository: "ggml-org/gpt-oss-120b-GGUF",
        revision: "f59990363fbbb67f01bce03e5e7bf0929e3e320b", file: "gpt-oss-120b-MXFP4.gguf",
        sha256: "582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d", bytes: 63_387_346_208,
        offload: Offload(moeLayers: 36, ncmoeSweep: [36, 28, 24], threadsSweep: [8, 16], safeNcmoe: 24),
        capabilities: Capabilities(inputs: [.text], toolCalls: true,
            thinking: [.reasoningEffort], contextLength: 131_072, knowledgeCutoff: "2024-06")),
    Model(name: "gpt-oss-20b", purposes: [.general],
        note: "gpt-oss 20B: MoE, 3.6B active; the service's model",
        repository: "ggml-org/gpt-oss-20b-GGUF",
        revision: "15dcfc7e7c608c6161507211ed43c5b590801b4b", file: "gpt-oss-20b-MXFP4.gguf",
        sha256: "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901", bytes: 12_109_566_624,
        offload: nil,
        capabilities: Capabilities(inputs: [.text], toolCalls: true,
            thinking: [.reasoningEffort], contextLength: 131_072, knowledgeCutoff: "2024-06")),
]

// Selection order: by primary purpose (coding, prose, general), then as declared.
let models: [Model] = declaredModels.enumerated()
    .sorted { ($0.element.primary, $0.offset) < ($1.element.primary, $1.offset) }
    .map(\.element)

enum Suite: String, CaseIterable {
    case baseline, depth, offload, reuse, state

    var summary: String {
        switch self {
        case .baseline: return "pp512 and tg128 at empty context"
        case .depth: return "pp512 and tg128 at 0, 8k, 32k and 64k tokens of context"
        case .offload: return "--n-cpu-moe x threads sweep (models too large for VRAM)"
        case .reuse: return "real tasks through tools; --store <sqlite> turns the reuse store on"
        case .state: return "KV-cache state store against an out-of-memory error: runs 1-4 at --depth (--store <sqlite>)"
        }
    }
}

let depths = [0, 8192, 32768, 65536]

func benchArguments(_ model: Model, _ suite: Suite, ncmoe: [Int]?, threads: [Int]?, kv: String?, reps: Int) -> [String] {
    var argv = onAccelerator("llama-bench") + ["-m", model.path, "-fa", "on", "-r", "\(reps)", "-o", "jsonl", "-oe", "md"]
    // Quantized V needs flash attention, which every suite turns on.
    if let kv { argv += ["-ctk", kv, "-ctv", kv] }
    if let ubatch { argv += ["-ub", "\(ubatch)"] }
    let list: ([Int]) -> String = { $0.map(String.init).joined(separator: ",") }
    switch suite {
    case .baseline, .depth:
        if suite == .depth { argv += ["-d", list(depths)] }
        // Expert offload is for the dGPU's 24 GB; the iGPU's memory is system
        // RAM, so there the model is whole on the device unless --ncmoe says otherwise.
        if let ncmoe {
            argv += ["-ncmoe", list(ncmoe)]
        } else if let offload = model.offload, accelerator.usesDGPU {
            argv += ["-ncmoe", "\(offload.safeNcmoe)"]
        }
        if let threads {
            argv += ["-t", list(threads)]
        } else if accelerator.cpuOnly {
            argv += ["-t", list(cpuThreadsSweep)]
        }
    case .reuse:
        break  // runs llama-server, not llama-bench (runReuse)
    case .state:
        break  // five llama-bench runs of its own (runStateStore)
    case .offload:
        guard let offload = model.offload else {
            fatal("\(model.name) fits in VRAM whole; the offload suite is for models declared with an offload.")
        }
        guard accelerator.usesDGPU else {
            fatal("the offload suite sweeps expert offload from the dGPU's VRAM; \(accelerator.name) holds the model whole.")
        }
        argv += ["-ncmoe", list(ncmoe ?? offload.ncmoeSweep), "-t", list(threads ?? offload.threadsSweep)]
    }
    return argv
}

// MARK: - State

func fileSize(_ path: String) -> Int64? {
    (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?.int64Value
}

func gpuComputePids() -> [String] {
    query(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]).output
        .split(separator: "\n").map { trimmed(String($0)) }.filter { !$0.isEmpty }
}

func hostFacts() -> [(String, String)] {
    let read: (String) -> String = { trimmed((try? String(contentsOfFile: $0, encoding: .utf8)) ?? "?") }
    let cpu = read("/proc/cpuinfo").split(separator: "\n").first { $0.hasPrefix("model name") }
        .map { trimmed(String($0.split(separator: ":", maxSplits: 1).last ?? "")) } ?? "?"
    let udev = query(["udevadm", "info", "-e"]).output.split(separator: "\n")
    let speeds = udev.filter { $0.contains("_CONFIGURED_SPEED_MTS=") }.map { $0.split(separator: "=").last.map(String.init) ?? "?" }
    // MEMORY_DEVICE_<n>_SIZE only: udev also lists MEMORY_DEVICE_<n>_VOLATILE_SIZE
    // for each module, which counted every module twice.
    let sizes = udev.compactMap { line -> Int64? in
        guard let range = line.range(of: "MEMORY_DEVICE_") else { return nil }
        let parts = line[range.upperBound...].split(separator: "=", maxSplits: 1)
        guard parts.count == 2, case let key = parts[0].split(separator: "_"), key.count == 2,
            Int(key[0]) != nil, key[1] == "SIZE" else { return nil }
        return Int64(parts[1])
    }
    let memory = sizes.isEmpty ? "?"
        : "\(sizes.count) x \(sizes[0] / 1_073_741_824) GB at \(Set(speeds).sorted().joined(separator: "/")) MT/s"
    // nvidia-smi only where the run uses the dGPU anyway: it wakes it.
    let gpu = accelerator.usesDGPU
        ? trimmed(query(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).output) : ""
    let build = buildVersion()
    return [
        ("Host", "\(read("/sys/class/dmi/id/sys_vendor")) \(read("/sys/class/dmi/id/product_version"))"),
        ("CPU", cpu),
        ("Memory", memory),
        ("Accelerator", buildDirectory == nil ? "\(accelerator.name): \(accelerator.summary) (llama-on \(accelerator.name))" : "\(accelerator.summary)"),
        ("dGPU, driver", accelerator.usesDGPU ? (gpu.isEmpty ? "?" : gpu) : "not used (left asleep; not queried)"),
        ("Kernel", trimmed(query(["uname", "-r"]).output)),
        ("llama.cpp", build.isEmpty ? "?" : build),
    ]
}

// MARK: - GPU telemetry during a run

let gpuFields = [
    "timestamp", "power.draw", "clocks.current.sm", "clocks.current.memory", "temperature.gpu",
    "utilization.gpu", "memory.used", "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.sw_thermal_slowdown", "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
]

let gpuLimitNames = ["software power cap", "software thermal slowdown", "hardware thermal slowdown",
    "hardware power brake"]

// A thread in this process runs one nvidia-smi query per tick. It is not
// nvidia-smi's own loop (-lms): that child outlives a benchmark killed
// outright, re-parents to the user manager, and polls forever, keeping the
// dGPU awake (six such orphans held it on 2026-11-13). A thread dies with
// the process, however it ends; at most one query is in flight.
final class GpuSampler {
    private let csv: String
    private var stopped = false
    private let lock = NSLock()
    private let done = DispatchSemaphore(value: 0)

    init?(_ csv: String) {
        self.csv = csv
        guard FileManager.default.createFile(atPath: csv,
            contents: Data((gpuFields.joined(separator: ", ") + "\n").utf8)) else { return nil }
        let thread = Thread { [self] in run() }
        thread.start()
    }

    private var isStopped: Bool {
        lock.lock()
        defer { lock.unlock() }
        return stopped
    }

    private func run() {
        let argv = ["nvidia-smi", "--query-gpu=\(gpuFields.joined(separator: ","))", "--format=csv,noheader,nounits"]
        while !isStopped {
            let tick = Date()
            let sample = query(argv)
            if sample.succeeded, let handle = FileHandle(forWritingAtPath: csv) {
                handle.seekToEndOfFile()
                handle.write(Data(sample.output.utf8))
                try? handle.close()
            }
            let rest = 0.5 - Date().timeIntervalSince(tick)
            if rest > 0 { usleep(useconds_t(rest * 1_000_000)) }
        }
        done.signal()
    }

    func stop() {
        lock.lock()
        stopped = true
        lock.unlock()
        done.wait()
    }
}

func startGpuSampler(_ csv: String) -> GpuSampler? { GpuSampler(csv) }

func stopGpuSampler(_ sampler: GpuSampler?) { sampler?.stop() }

// Summary over the samples under load (utilization at least 50%), so the
// load and unload phases do not dilute it.
func gpuSection(_ csv: String) -> String {
    let rows = ((try? String(contentsOfFile: csv, encoding: .utf8)) ?? "").split(separator: "\n").dropFirst()
        .map { $0.split(separator: ",", omittingEmptySubsequences: false).map { trimmed(String($0)) } }
        .filter { $0.count == gpuFields.count }
    let name = (csv as NSString).lastPathComponent
    var lines = ["", "## GPU during the run", ""]
    let loaded = rows.filter { (Double($0[5]) ?? 0) >= 50 }
    lines.append("Sampled every 0.5 s by nvidia-smi into `\(name)`: \(rows.count) samples, \(loaded.count) under load "
        + "(utilization at least 50%).")
    guard !loaded.isEmpty else { return (lines + ["", "No samples under load."]).joined(separator: "\n") + "\n" }
    func values(_ column: Int) -> [Double] { loaded.compactMap { Double($0[column]) } }
    func stat(_ column: Int, _ format: String) -> (avg: String, min: String, max: String) {
        let v = values(column)
        guard !v.isEmpty else { return ("?", "?", "?") }
        return (String(format: format, v.reduce(0, +) / Double(v.count)), String(format: format, v.min()!),
            String(format: format, v.max()!))
    }
    let power = stat(1, "%.0f"), sm = stat(2, "%.0f"), memory = stat(3, "%.0f"), temperature = stat(4, "%.0f"),
        vram = stat(6, "%.0f")
    let limits = gpuLimitNames.enumerated().compactMap { index, label -> String? in
        let active = loaded.filter { $0[7 + index] == "Active" }.count
        return active == 0 ? nil : "\(label) \(active * 100 / loaded.count)%"
    }
    lines += [
        "",
        "| Under load | Average | Min | Max |",
        "|---|---:|---:|---:|",
        "| Power, W | \(power.avg) | \(power.min) | \(power.max) |",
        "| SM clock, MHz | \(sm.avg) | \(sm.min) | \(sm.max) |",
        "| Memory clock, MHz | \(memory.avg) | \(memory.min) | \(memory.max) |",
        "| Temperature, °C | \(temperature.avg) | \(temperature.min) | \(temperature.max) |",
        "| VRAM used, MiB | \(vram.avg) | \(vram.min) | \(vram.max) |",
        "",
        "Clock limits (share of samples under load): \(limits.isEmpty ? "none" : limits.joined(separator: ", ")).",
    ]
    return lines.joined(separator: "\n") + "\n"
}

func appendGpuSection(toReport path: String, csv: String) {
    guard let report = try? String(contentsOfFile: path, encoding: .utf8) else { return }
    try? (report + gpuSection(csv)).write(toFile: path, atomically: true, encoding: .utf8)
}

// MARK: - Report

func number(_ value: Any?) -> Double? {
    if let value = value as? NSNumber { return value.doubleValue }
    if let value = value as? String { return Double(value) }
    return nil
}

func testName(_ row: [String: Any]) -> String {
    let prompt = Int(number(row["n_prompt"]) ?? 0)
    let gen = Int(number(row["n_gen"]) ?? 0)
    let depth = Int(number(row["n_depth"]) ?? 0)
    let base = prompt > 0 && gen > 0 ? "pp\(prompt)+tg\(gen)" : prompt > 0 ? "pp\(prompt)" : "tg\(gen)"
    return depth > 0 ? "\(base) @ d\(depth)" : base
}

func report(jsonl: String, model: Model, suite: Suite, argv: [String], started: Date, facts: [(String, String)]) -> String {
    let rows = ((try? String(contentsOfFile: jsonl, encoding: .utf8)) ?? "").split(separator: "\n")
        .compactMap { (try? JSONSerialization.jsonObject(with: Data($0.utf8))) as? [String: Any] }
    let stamp = ISO8601DateFormatter().string(from: started)
    var lines = [
        "# llama-bench: \(model.name), \(suite.rawValue), on \(accelerator.name)",
        "",
        "| | |",
        "|---|---|",
        "| Run (UTC) | \(stamp) |",
        "| Model | `\(model.file)` (\(model.repository) @ \(model.revision.prefix(12)), sha256 \(model.sha256.prefix(12))…) |",
        "| Suite | \(suite.rawValue): \(suite.summary) |",
    ]
    lines += facts.map { "| \($0.0) | \($0.1) |" }
    // What llama-bench itself saw: its backends and GPUs.
    if let first = rows.first {
        let backends = first["backends"] as? String ?? "?"
        let gpus = (first["gpu_info"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "none"
        lines.append("| Devices (llama-bench) | \(backends); GPUs: \(gpus) |")
    }
    lines += [
        "| Command | `\(argv.joined(separator: " "))` |",
        "",
        "| Test | n-cpu-moe | Threads | KV cache | t/s | ± |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in rows {
        let ts = number(row["avg_ts"]).map { String(format: "%.2f", $0) } ?? "?"
        let sd = number(row["stddev_ts"]).map { String(format: "%.2f", $0) } ?? "?"
        let ncmoe = number(row["n_cpu_moe"]).map { String(Int($0)) } ?? "?"
        let threads = number(row["n_threads"]).map { String(Int($0)) } ?? "?"
        let cache = (row["type_k"] as? String).map { k in (row["type_v"] as? String).map { $0 == k ? k : "\(k)/\($0)" } ?? k } ?? "?"
        lines.append("| \(testName(row)) | \(ncmoe) | \(threads) | \(cache) | \(ts) | \(sd) |")
    }
    if rows.isEmpty { lines.append("| (no results) | | | | | |") }
    return lines.joined(separator: "\n") + "\n"
}

// MARK: - Reuse suite
//
// Real tasks, not synthetic tokens: a model answers questions about a source
// tree through tools, with the reuse store on (--store) or off. What it
// measures is what the store is for: tokens, peak context, prompt time, and
// whether the work fits the context at all.

var llamaServer: String { buildDirectory.map { "\($0)/llama-server" } ?? "/opt/llama.cpp/current/bin/llama-server" }
let reusePort = 9940
let reuseBase = "http://127.0.0.1:\(reusePort)"
let maxTurns = 24
// Output budget per turn. 4096 cut off long answers from thinking models (calibration, 2026-11-13);
// a turn that still reaches it is recorded as "output limit", never as an answer.
let maxTurnTokens = 16384
let readCap = 32_000          // characters per read_file call
let scriptDirectory = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.deletingLastPathComponent().path

var storePath: String?
var reuseContext = 32_768
var reuseRepeats = 1
var roleFilter: String?
// Calibration threshold: reports count the tasks whose peak context went over this.
// A test threshold, not a hardware limit: llama-server runs Gemma 4 31B at 65,536
// tokens of context on 24 GB (see llama-bench-oom-reuse-sqlite.md).
var qualifyAbove = 32_768
var tasksPath: String?
var sourcePath: String?

// stdin in, stdout and stderr merged out (sqlite3 reads its SQL from stdin).
func queryInput(_ argv: [String], input: String) -> Captured {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = argv
    let inPipe = Pipe()
    let outPipe = Pipe()
    process.standardInput = inPipe
    process.standardOutput = outPipe
    process.standardError = outPipe
    do { try process.run() } catch { return Captured(status: 127, output: "") }
    let data = Data(input.utf8)
    let writer = inPipe.fileHandleForWriting
    DispatchQueue.global().async {
        writer.write(data)
        try? writer.close()
    }
    let out = outPipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return Captured(status: process.terminationStatus, output: String(decoding: out, as: UTF8.self))
}

func sqlText(_ text: String) -> String {
    "'" + text.replacingOccurrences(of: "\u{0}", with: "").replacingOccurrences(of: "'", with: "''") + "'"
}

func jsonObject(_ text: String) -> [String: Any]? {
    (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any]
}

func jsonString(_ value: Any) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) else { return "null" }
    return String(decoding: data, as: UTF8.self)
}

// The store: one SQLite file the user owns. Entries are append-only (triggers
// refuse UPDATE and DELETE); an entry is valid while every file it depends on
// still has the SHA-256 recorded with it.
struct Store {
    let path: String

    static let schema = """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY, key TEXT NOT NULL, kind TEXT NOT NULL, content TEXT NOT NULL,
            depends TEXT NOT NULL, model TEXT NOT NULL, run TEXT NOT NULL, task TEXT NOT NULL,
            tokens_spent INTEGER NOT NULL, created TEXT NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(key, content, content='entries', content_rowid='id');
        CREATE TRIGGER IF NOT EXISTS entries_index AFTER INSERT ON entries BEGIN
            INSERT INTO entries_fts(rowid, key, content) VALUES (new.id, new.key, new.content); END;
        CREATE TRIGGER IF NOT EXISTS entries_no_update BEFORE UPDATE ON entries BEGIN
            SELECT RAISE(ABORT, 'entries are append-only'); END;
        CREATE TRIGGER IF NOT EXISTS entries_no_delete BEFORE DELETE ON entries BEGIN
            SELECT RAISE(ABORT, 'entries are append-only'); END;
        CREATE TABLE IF NOT EXISTS hits (
            id INTEGER PRIMARY KEY, entry INTEGER NOT NULL REFERENCES entries(id),
            run TEXT NOT NULL, task TEXT NOT NULL, at TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS hits_no_update BEFORE UPDATE ON hits BEGIN
            SELECT RAISE(ABORT, 'hits are append-only'); END;
        CREATE TRIGGER IF NOT EXISTS hits_no_delete BEFORE DELETE ON hits BEGIN
            SELECT RAISE(ABORT, 'hits are append-only'); END;
        """

    @discardableResult
    func exec(_ sql: String) -> Captured { queryInput(["sqlite3", "-bail", path], input: sql) }

    func rows(_ sql: String) -> [[String: Any]] {
        let result = queryInput(["sqlite3", "-bail", "-json", path], input: sql)
        guard result.succeeded, !trimmed(result.output).isEmpty else { return [] }
        return ((try? JSONSerialization.jsonObject(with: Data(result.output.utf8))) as? [[String: Any]]) ?? []
    }

    func open() {
        let directory = (path as NSString).deletingLastPathComponent
        do { try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true) } catch {
            fatal("cannot create \(directory): \(error)")
        }
        let result = exec(Store.schema)
        if !result.succeeded { fatal("cannot open the store \(path): \(trimmed(result.output))") }
    }

    var entryCount: Int { Int(number(rows("SELECT count(*) AS n FROM entries;").first?["n"]) ?? 0) }
}

struct ReuseTask {
    let id: String
    let question: String
    let expect: [[String]]
    let caseSensitive: Bool
    let tag: String        // optional overlap tag; the report totals by it
    let role: String       // "measured" (default) or "build": build tasks only fill a store
}

struct TaskSet {
    let name: String
    let path: String
    let source: String
    let authoredAgainst: String
    let tasks: [ReuseTask]
}

func loadTaskSet() -> TaskSet {
    let path = tasksPath ?? "\(scriptDirectory)/reuse-tasks-c++.json"
    guard let data = FileManager.default.contents(atPath: path),
        let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
        let items = root["tasks"] as? [[String: Any]]
    else { fatal("cannot read the task set \(path) (--tasks names another).") }
    let tasks = items.compactMap { item -> ReuseTask? in
        guard let id = item["id"] as? String, let question = item["question"] as? String,
            let expect = item["expect"] as? [[String]], !expect.isEmpty
        else { return nil }
        return ReuseTask(id: id, question: question, expect: expect, caseSensitive: item["caseSensitive"] as? Bool ?? false,
            tag: item["tag"] as? String ?? "", role: item["role"] as? String ?? "measured")
    }
    if tasks.count != items.count { fatal("\(path): every task needs id, question and a non-empty expect.") }
    // The source is --source, or the first of the task set's candidates (paths
    // relative to the task file) that exists, so identical copies of the task
    // set work from each repository they live in.
    let base = URL(fileURLWithPath: (path as NSString).deletingLastPathComponent)
    let declared = (root["source"] as? [String] ?? (root["source"] as? String).map { [$0] } ?? [])
        .map { URL(fileURLWithPath: $0, relativeTo: base).standardizedFileURL.path }
    let found = sourcePath.map { [$0] } ?? declared
    guard let source = found.first(where: { FileManager.default.fileExists(atPath: $0) }) else {
        fatal("no source tree; looked for:", found.isEmpty ? "(none declared)" : found.joined(separator: ", "),
            "Pass --source <dir>: the sound-design checkout's enforcement/languages/programming/c++.")
    }
    // --role keeps only the measured or only the build tasks.
    let chosen = roleFilter.map { role in tasks.filter { $0.role == role } } ?? tasks
    if chosen.isEmpty { fatal("\(path) has no \(roleFilter ?? "") tasks.") }
    let name = (root["name"] as? String ?? (path as NSString).lastPathComponent) + (roleFilter.map { " (\($0) tasks)" } ?? "")
    return TaskSet(name: name, path: path, source: URL(fileURLWithPath: source).standardizedFileURL.path,
        authoredAgainst: root["authoredAgainst"] as? String ?? "?", tasks: chosen)
}

// A path inside the source tree, or nil.
func inside(_ relative: String, _ root: String) -> String? {
    let resolved = URL(fileURLWithPath: root).appendingPathComponent(relative).standardizedFileURL.path
    return resolved == root || resolved.hasPrefix(root + "/") ? resolved : nil
}

var hashCache: [String: String] = [:]
func sha256(_ path: String) -> String? {
    if let known = hashCache[path] { return known }
    let digest = query(["sha256sum", path]).output.split(separator: " ").first.map(String.init)
    if let digest, digest.count == 64 { hashCache[path] = digest }
    return hashCache[path]
}

// MARK: Tools

let fileTools: [[String: Any]] = [
    ["type": "function", "function": [
        "name": "list_files",
        "description": "List files under a directory of the source tree, with sizes in bytes.",
        "parameters": ["type": "object", "properties": [
            "path": ["type": "string", "description": "Directory relative to the source root; default the root"]]],
    ]],
    ["type": "function", "function": [
        "name": "read_file",
        "description": "Read a file of the source tree, optionally a line range. Long files are truncated; ask for later lines by range.",
        "parameters": ["type": "object", "properties": [
            "path": ["type": "string"], "start_line": ["type": "integer"], "end_line": ["type": "integer"]],
            "required": ["path"]],
    ]],
    ["type": "function", "function": [
        "name": "grep",
        "description": "Search the source tree for an extended regular expression; returns file:line:text matches.",
        "parameters": ["type": "object", "properties": [
            "pattern": ["type": "string"], "path": ["type": "string", "description": "Directory or file; default the root"]],
            "required": ["pattern"]],
    ]],
]

let storeTools: [[String: Any]] = [
    ["type": "function", "function": [
        "name": "store_search",
        "description": "Search the reuse store for work already done: summaries, answers, extracted facts. Returns only entries whose source files are unchanged.",
        "parameters": ["type": "object", "properties": ["query": ["type": "string"]], "required": ["query"]],
    ]],
    ["type": "function", "function": [
        "name": "store_lookup",
        "description": "Fetch one reuse-store entry by its exact key, if it is still valid.",
        "parameters": ["type": "object", "properties": ["key": ["type": "string"]], "required": ["key"]],
    ]],
    ["type": "function", "function": [
        "name": "store_put",
        "description": "Save something you derived so later work need not recompute it. depends_on lists the source files it was derived from.",
        "parameters": ["type": "object", "properties": [
            "key": ["type": "string", "description": "Descriptive and stable, e.g. 'summary: environment/README.md'"],
            "kind": ["type": "string", "enum": ["summary", "answer", "structure", "decision"]],
            "content": ["type": "string"],
            "depends_on": ["type": "array", "items": ["type": "string"]]],
            "required": ["key", "kind", "content", "depends_on"]],
    ]],
]

let basePrompt = """
    You answer questions about the files in a source tree, using the tools to read them. \
    Read only what the question needs: search before reading whole files, and read line ranges of long files. \
    Answer concisely and name the exact identifiers, flags, files and values as the files write them.
    """

let storePrompt = """

    A reuse store holds work already done on this tree: summaries, answers and extracted facts, each valid \
    while the files it depends on are unchanged. Before reading or searching files, call store_search with the \
    key terms of the question; use a valid entry instead of re-reading. After you derive something another \
    question could reuse, such as a summary of a file or section, a list of facts from it, or your answer, \
    call store_put with a descriptive key, the content, and every file it depends on.
    """

struct ReuseContext {
    let store: Store?
    let root: String
    let run: String
    let model: String
}

func listFiles(_ arguments: [String: Any], _ context: ReuseContext) -> String {
    guard let directory = inside(arguments["path"] as? String ?? ".", context.root) else { return "error: path outside the source tree" }
    guard let walker = FileManager.default.enumerator(atPath: directory) else { return "error: no such directory" }
    var lines: [String] = []
    while let item = walker.nextObject() as? String {
        if item == ".git" || item.hasPrefix(".git/") { walker.skipDescendants(); continue }
        let full = "\(directory)/\(item)"
        var isDirectory: ObjCBool = false
        if !FileManager.default.fileExists(atPath: full, isDirectory: &isDirectory) || isDirectory.boolValue { continue }
        lines.append("\(String(full.dropFirst(context.root.count + 1))) (\(fileSize(full) ?? 0) bytes)")
        if lines.count >= 400 { lines.append("[listing truncated at 400 files]"); break }
    }
    return lines.isEmpty ? "(no files)" : lines.joined(separator: "\n")
}

func readFile(_ arguments: [String: Any], _ context: ReuseContext) -> String {
    guard let relative = arguments["path"] as? String, let path = inside(relative, context.root) else {
        return "error: path missing or outside the source tree"
    }
    guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return "error: cannot read \(relative)" }
    let lines = text.components(separatedBy: "\n")
    let start = max(1, Int(number(arguments["start_line"]) ?? 1))
    let end = min(lines.count, Int(number(arguments["end_line"]) ?? Double(lines.count)))
    guard start <= end else { return "error: empty range; the file has \(lines.count) lines" }
    var out = ""
    for index in start...end {
        let line = "\(index)\t\(lines[index - 1])\n"
        if out.count + line.count > readCap {
            return out + "[truncated at line \(index - 1) of \(lines.count); read on with start_line=\(index)]"
        }
        out += line
    }
    return out
}

func grepTree(_ arguments: [String: Any], _ context: ReuseContext) -> String {
    guard let pattern = arguments["pattern"] as? String, !pattern.isEmpty else { return "error: pattern missing" }
    guard let target = inside(arguments["path"] as? String ?? ".", context.root) else { return "error: path outside the source tree" }
    let result = query(["grep", "-rnIE", "--exclude-dir=.git", "--max-count=20", "-e", pattern, "--", target])
    let lines = result.output.split(separator: "\n").map { line -> String in
        line.hasPrefix(context.root + "/") ? String(line.dropFirst(context.root.count + 1)) : String(line)
    }
    if lines.isEmpty { return result.status == 1 ? "(no matches)" : "error: grep failed (bad pattern?)" }
    return (lines.prefix(150) + (lines.count > 150 ? ["[\(lines.count - 150) more matches]"] : [])).joined(separator: "\n")
}

// Entries whose recorded file hashes still match the tree.
func validEntries(_ rows: [[String: Any]], _ context: ReuseContext) -> [[String: Any]] {
    rows.filter { row in
        guard let depends = (try? JSONSerialization.jsonObject(with: Data((row["depends"] as? String ?? "[]").utf8))) as? [[String: String]]
        else { return false }
        return depends.allSatisfy { item in
            guard let file = item["path"], let hash = item["sha256"], let path = inside(file, context.root) else { return false }
            return sha256(path) == hash
        }
    }
}

func recordHits(_ entries: [[String: Any]], _ context: ReuseContext, task: String) {
    guard let store = context.store, !entries.isEmpty else { return }
    let now = ISO8601DateFormatter().string(from: Date())
    let values = entries.compactMap { number($0["id"]) }.map {
        "(\(Int($0)), \(sqlText(context.run)), \(sqlText(task)), \(sqlText(now)))"
    }
    store.exec("INSERT INTO hits(entry, run, task, at) VALUES \(values.joined(separator: ", "));")
}

func dependsPaths(_ entries: [[String: Any]]) -> Set<String> {
    var paths = Set<String>()
    for row in entries {
        let items = (try? JSONSerialization.jsonObject(with: Data((row["depends"] as? String ?? "[]").utf8))) as? [[String: String]] ?? []
        for item in items { if let path = item["path"] { paths.insert(path) } }
    }
    return paths
}

func render(_ entries: [[String: Any]]) -> String {
    entries.map { "key: \($0["key"] as? String ?? "")\nkind: \($0["kind"] as? String ?? "")\n\($0["content"] as? String ?? "")" }
        .joined(separator: "\n---\n")
}

// Newest valid entry wins for a key; superseded entries stay (append-only).
func storeLookup(_ arguments: [String: Any], _ context: ReuseContext, task: String) -> (String, [[String: Any]]) {
    guard let store = context.store, let key = arguments["key"] as? String else { return ("error: key missing", []) }
    let rows = store.rows("SELECT id, key, kind, content, depends FROM entries WHERE key = \(sqlText(key)) ORDER BY id DESC;")
    guard let entry = validEntries(rows, context).first else { return (rows.isEmpty ? "miss" : "stale: its source files changed", []) }
    recordHits([entry], context, task: task)
    return (render([entry]), [entry])
}

func storeSearch(_ arguments: [String: Any], _ context: ReuseContext, task: String) -> (String, [[String: Any]]) {
    guard let store = context.store, let text = arguments["query"] as? String else { return ("error: query missing", []) }
    // Words only, each quoted, OR-ed: model text never reaches FTS5 syntax.
    let words = text.split { !($0.isLetter || $0.isNumber || $0 == "_") }.map(String.init).filter { $0.count > 1 }
    guard !words.isEmpty else { return ("miss", []) }
    let match = words.prefix(16).map { "\"\($0.replacingOccurrences(of: "\"", with: ""))\"" }.joined(separator: " OR ")
    let rows = store.rows("""
        SELECT e.id, e.key, e.kind, e.content, e.depends FROM entries_fts f JOIN entries e ON e.id = f.rowid
        WHERE entries_fts MATCH \(sqlText(match)) ORDER BY bm25(entries_fts) LIMIT 20;
        """)
    var seen = Set<String>()
    let valid = validEntries(rows, context).filter { seen.insert($0["key"] as? String ?? "").inserted }.prefix(5)
    guard !valid.isEmpty else { return ("miss", []) }
    recordHits(Array(valid), context, task: task)
    return (render(Array(valid)), Array(valid))
}

func storePut(_ arguments: [String: Any], _ context: ReuseContext, task: String, tokensSpent: Int) -> String {
    // depends_on paths are relative to the source root.
    guard let store = context.store, let key = arguments["key"] as? String, let content = arguments["content"] as? String,
        !key.isEmpty, !content.isEmpty
    else { return "error: key and content are required" }
    let kind = arguments["kind"] as? String ?? "summary"
    var depends: [[String: String]] = []
    for file in arguments["depends_on"] as? [String] ?? [] {
        guard let path = inside(file, context.root), let hash = sha256(path) else { return "error: depends_on file not found: \(file)" }
        depends.append(["path": String(path.dropFirst(context.root.count + 1)), "sha256": hash])
    }
    if depends.isEmpty { return "error: depends_on must name at least one source file" }
    let result = store.exec("""
        INSERT INTO entries(key, kind, content, depends, model, run, task, tokens_spent, created) VALUES (
        \(sqlText(key)), \(sqlText(kind)), \(sqlText(content)), \(sqlText(jsonString(depends))), \(sqlText(context.model)),
        \(sqlText(context.run)), \(sqlText(task)), \(tokensSpent), \(sqlText(ISO8601DateFormatter().string(from: Date()))));
        """)
    return result.succeeded ? "stored" : "error: \(trimmed(result.output))"
}

// MARK: Server and loop

func startServer(_ model: Model, ncmoe: [Int]?, kv: String?, log: String) -> Process? {
    var argv = onAccelerator("llama-server") + ["-m", model.path, "--host", "127.0.0.1", "--port", "\(reusePort)",
        "--ctx-size", "\(reuseContext)", "--parallel", "1", "--n-gpu-layers", "all", "--flash-attn", "on",
        "--offline", "--no-ui"]
    if let kv { argv += ["-ctk", kv, "-ctv", kv] }
    if let ubatch { argv += ["--ubatch-size", "\(ubatch)"] }
    if let ncmoe {
        argv += ["--n-cpu-moe", "\(ncmoe[0])"]
    } else if let offload = model.offload, accelerator.usesDGPU {
        argv += ["--n-cpu-moe", "\(offload.safeNcmoe)"]
    }
    print("    >>> \(argv.joined(separator: " "))  (log: \(log))")
    _ = FileManager.default.createFile(atPath: log, contents: nil)
    guard let handle = FileHandle(forWritingAtPath: log) else { fatal("cannot write \(log)") }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: argv[0])
    process.arguments = Array(argv.dropFirst())
    process.standardOutput = handle
    process.standardError = handle
    do { try process.run() } catch { fatal("cannot start \(argv.joined(separator: " ")): \(error)") }
    let deadline = Date().addingTimeInterval(300)
    while Date() < deadline && process.isRunning {
        if query(["curl", "-fsS", "--max-time", "2", "\(reuseBase)/health"]).succeeded { return process }
        usleep(1_000_000)
    }
    if process.isRunning { process.terminate(); process.waitUntilExit() }
    return nil
}

func stopServer(_ process: Process) {
    process.terminate()
    let deadline = Date().addingTimeInterval(20)
    while process.isRunning && Date() < deadline { usleep(200_000) }
    if process.isRunning { kill(process.processIdentifier, SIGKILL) }
    process.waitUntilExit()
}

struct TaskResult {
    let id: String
    var tag = ""                // the task set's overlap tag, if any (same-section, same-file, new-file)
    var role = "measured"       // or "build"
    var repeatIndex = 1
    var outcome = "unfinished"
    var correct = false
    var missing: [String] = []
    var turns = 0
    var peakContext = 0         // the largest context a turn the server accepted held: prompt + generated
    var refusedRequest = 0      // a request the server refused as over the context: its size (not a peak)
    var promptSent = 0          // usage.prompt_tokens, summed: what the model was given
    var promptProcessed = 0     // timings.prompt_n, summed: what it computed (the rest was cached)
    var promptSeconds = 0.0
    var generated = 0
    var generateSeconds = 0.0
    var wallSeconds = 0.0
    var toolCalls: [String: Int] = [:]
    var storeHits = 0           // entries the model fetched with store tools
    var storePuts = 0           // entries the model saved
    var harnessHits = 0         // entries the harness put in front of the model before it started
    var harnessPut = false      // the harness stored the answer
    var fromStore = false       // P4: answered with no file tool call, from store entries alone
    var served: [[String: Any]] = []   // P3: every entry offered, with how it was reached
    var answer = ""

    var asJSON: [String: Any] {
        ["id": id, "tag": tag, "role": role, "repeat": repeatIndex, "outcome": outcome, "correct": correct, "missing": missing,
            "turns": turns, "peak_context": peakContext, "refused_request": refusedRequest, "prompt_sent": promptSent, "prompt_processed": promptProcessed,
            "prompt_seconds": promptSeconds, "generated": generated, "generate_seconds": generateSeconds,
            "wall_seconds": wallSeconds, "tool_calls": toolCalls, "store_hits": storeHits,
            "store_puts": storePuts, "harness_hits": harnessHits, "harness_put": harnessPut,
            "from_store": fromStore, "served": served, "answer": answer]
    }
}

func score(_ answer: String, _ task: ReuseTask) -> [String] {
    let text = task.caseSensitive ? answer : answer.lowercased()
    return task.expect.filter { group in
        !group.contains { text.contains(task.caseSensitive ? $0 : $0.lowercased()) }
    }.map { $0.joined(separator: " | ") }
}

// One request; the body goes through a file (it outgrows argv). Keys are
// sorted: Foundation's serializer on Linux does not keep key order (one body
// came out 4 different ways in 20 tries), so the tool definitions rendered
// differently near the top of every prompt and llama-server could reuse only
// ~100 tokens of its cache. Sorted keys make equal bodies equal bytes.
func chat(_ body: [String: Any], scratch: String) -> (Int, [String: Any]?, String) {
    let request = "\(scratch)/request.json"
    let response = "\(scratch)/response.json"
    guard let data = try? JSONSerialization.data(withJSONObject: body, options: [.sortedKeys]) else {
        return (0, nil, "cannot encode the request")
    }
    guard FileManager.default.createFile(atPath: request, contents: data) else { return (0, nil, "cannot write \(request)") }
    let status = Int(trimmed(query(["curl", "-sS", "--max-time", "900", "-H", "Content-Type: application/json",
        "--data-binary", "@\(request)", "-o", response, "-w", "%{http_code}", "\(reuseBase)/v1/chat/completions"]).output)) ?? 0
    let text = (try? String(contentsOfFile: response, encoding: .utf8)) ?? ""
    return (status, jsonObject(text), text)
}

// P3: the entries a task was offered, with how it reached them.
func servedRecords(_ entries: [[String: Any]], via: String) -> [[String: Any]] {
    entries.map {
        ["id": Int(number($0["id"]) ?? 0), "kind": $0["kind"] as? String ?? "", "via": via,
            "key": String(($0["key"] as? String ?? "").prefix(120))]
    }
}

// Lookup before compute, by the harness: the model does not have to choose
// to consult the store. After an answer, the harness stores it, depending on
// every file the model read or matched and every entry it reused, but only
// when the model read or matched a file (P1): an answer derived from entries
// alone adds nothing the store doesn't hold, and storing it duplicates them.
func runTask(_ task: ReuseTask, _ context: ReuseContext, server: Process, scratch: String) -> TaskResult {
    var result = TaskResult(id: task.id)
    result.tag = task.tag
    result.role = task.role
    let started = Date()
    let tools = fileTools + (context.store == nil ? [] : storeTools)
    var question = task.question
    var filesTouched = Set<String>()     // read or matched by the model's file tools
    var entryDepends = Set<String>()     // the files behind every entry it was offered
    if context.store != nil {
        let (text, entries) = storeSearch(["query": task.question], context, task: task.id)
        if !entries.isEmpty {
            result.harnessHits = entries.count
            result.served += servedRecords(entries, via: "harness")
            entryDepends.formUnion(dependsPaths(entries))
            question += "\n\nReuse-store entries for this question (valid: the files they derive from are unchanged). "
                + "If they answer the question, answer from them directly, without reading or searching files; "
                + "read files only for what they leave out.\n" + text
        }
    }
    var messages: [[String: Any]] = [
        ["role": "system", "content": basePrompt + (context.store == nil ? "" : storePrompt)],
        ["role": "user", "content": question],
    ]
    func finish(_ outcome: String) -> TaskResult {
        result.outcome = outcome
        result.wallSeconds = Date().timeIntervalSince(started)
        return result
    }
    for turn in 1...maxTurns {
        guard server.isRunning else { return finish("server stopped") }
        result.turns = turn
        let (status, json, text) = chat(["messages": messages, "tools": tools, "tool_choice": "auto",
            "temperature": 0, "seed": 1, "max_tokens": maxTurnTokens], scratch: scratch)
        guard status == 200, let json else {
            let error = jsonObject(text)?["error"] as? [String: Any]
            let detail = (error?["message"] as? String) ?? text
            // A request over the context is refused with its size. It never ran,
            // so it is not the peak: the peak stays the last turn that fit.
            if let refused = number(error?["n_prompt_tokens"]) {
                result.refusedRequest = Int(refused)
            }
            result.answer = String(detail.prefix(300))
            return finish(detail.lowercased().contains("context") ? "context exceeded" : "server error \(status)")
        }
        var turnTokens = 0
        if let usage = json["usage"] as? [String: Any] {
            let prompt = Int(number(usage["prompt_tokens"]) ?? 0)
            let completion = Int(number(usage["completion_tokens"]) ?? 0)
            result.promptSent += prompt
            turnTokens = prompt + completion
            result.peakContext = max(result.peakContext, turnTokens)
        }
        if let timings = json["timings"] as? [String: Any] {
            result.promptProcessed += Int(number(timings["prompt_n"]) ?? 0)
            result.promptSeconds += (number(timings["prompt_ms"]) ?? 0) / 1000
            result.generated += Int(number(timings["predicted_n"]) ?? 0)
            result.generateSeconds += (number(timings["predicted_ms"]) ?? 0) / 1000
        }
        let choice = (json["choices"] as? [[String: Any]])?.first ?? [:]
        let message = (choice["message"] as? [String: Any]) ?? [:]
        // A turn cut off by a limit is not an answer: its content is partial or
        // empty (a thinking model can spend the whole budget reasoning). Running
        // into the context is "context exceeded"; otherwise the turn budget ran out.
        if choice["finish_reason"] as? String == "length" {
            result.answer = trimmed(message["content"] as? String ?? "")
            result.missing = score(result.answer, task)
            return finish(turnTokens >= reuseContext - 8 ? "context exceeded" : "output limit")
        }
        let calls = message["tool_calls"] as? [[String: Any]] ?? []
        var assistant: [String: Any] = ["role": "assistant", "content": message["content"] as? String ?? ""]
        if !calls.isEmpty { assistant["tool_calls"] = calls }
        // Send the reasoning back with the turn that produced it, so the chat
        // template re-renders this turn as the tokens the model generated and
        // the next prompt continues the cached one. A hybrid model's recurrent
        // layers cannot be cut back to a divergence point inside the turn.
        if let reasoning = message["reasoning_content"] as? String, !reasoning.isEmpty {
            assistant["reasoning_content"] = reasoning
        }
        messages.append(assistant)
        if calls.isEmpty {
            result.answer = trimmed(message["content"] as? String ?? "")
            result.missing = score(result.answer, task)
            result.correct = result.missing.isEmpty
            result.fromStore = filesTouched.isEmpty && !result.served.isEmpty
            if context.store != nil && !result.answer.isEmpty && !filesTouched.isEmpty {
                let stored = storePut(["key": "answer: \(task.question)", "kind": "answer",
                    "content": "Question: \(task.question)\nAnswer: \(result.answer)",
                    "depends_on": Array(filesTouched.union(entryDepends)).sorted()],
                    context, task: task.id, tokensSpent: result.promptProcessed + result.generated)
                result.harnessPut = stored == "stored"
            }
            return finish("answered")
        }
        for call in calls {
            let function = call["function"] as? [String: Any] ?? [:]
            let name = function["name"] as? String ?? "?"
            let arguments = jsonObject(function["arguments"] as? String ?? "{}") ?? [:]
            result.toolCalls[name, default: 0] += 1
            let output: String
            switch name {
            case "list_files": output = listFiles(arguments, context)
            case "read_file":
                output = readFile(arguments, context)
                if let file = arguments["path"] as? String, !output.hasPrefix("error:"), let path = inside(file, context.root) {
                    filesTouched.insert(String(path.dropFirst(context.root.count + 1)))
                }
            case "grep":
                output = grepTree(arguments, context)
                for line in output.split(separator: "\n") {
                    if let colon = line.firstIndex(of: ":"), inside(String(line[..<colon]), context.root) != nil,
                        FileManager.default.fileExists(atPath: "\(context.root)/\(line[..<colon])")
                    {
                        filesTouched.insert(String(line[..<colon]))
                    }
                }
            case "store_search" where context.store != nil:
                let (text, entries) = storeSearch(arguments, context, task: task.id)
                result.storeHits += entries.count
                result.served += servedRecords(entries, via: "model")
                entryDepends.formUnion(dependsPaths(entries))
                output = text
            case "store_lookup" where context.store != nil:
                let (text, entries) = storeLookup(arguments, context, task: task.id)
                result.storeHits += entries.count
                result.served += servedRecords(entries, via: "model")
                entryDepends.formUnion(dependsPaths(entries))
                output = text
            case "store_put" where context.store != nil:
                output = storePut(arguments, context, task: task.id, tokensSpent: result.promptProcessed + result.generated)
                if output == "stored" { result.storePuts += 1 }
            default: output = "error: no tool named \(name)"
            }
            messages.append(["role": "tool", "tool_call_id": call["id"] as? String ?? "", "content": output])
        }
    }
    return finish("turn limit")
}

// MARK: Report

let f1: (Double) -> String = { String(format: "%.1f", $0) }

// The share of the prompt served from llama-server's cache. Near 0% on a
// multi-turn task means every turn recomputed the whole conversation.
func reuseShare(_ sent: Int, _ computed: Int) -> String {
    sent > 0 ? String(format: "%.0f%%", 100 * (1 - Double(computed) / Double(sent))) : "—"
}

// Mean and range of one measure over the repetitions of a task, e.g. "7,410 (6,102–8,488)".
// Formatted by hand: Linux Foundation's NumberFormatter ignores fraction digits.
func grouped(_ value: Double, decimals: Int) -> String {
    let text = String(format: "%.\(decimals)f", value)
    let parts = text.split(separator: ".", maxSplits: 1).map(String.init)
    let negative = parts[0].hasPrefix("-")
    let digits = Array(negative ? parts[0].dropFirst() : Substring(parts[0]))
    var integer = ""
    for (index, digit) in digits.enumerated() {
        if index > 0 && (digits.count - index) % 3 == 0 { integer += "," }
        integer.append(digit)
    }
    return (negative ? "-" : "") + integer + (parts.count > 1 ? "." + parts[1] : "")
}

func spread(_ values: [Double], decimals: Int = 0) -> String {
    guard let low = values.min(), let high = values.max(), !values.isEmpty else { return "—" }
    let mean = values.reduce(0, +) / Double(values.count)
    let format: (Double) -> String = { grouped($0, decimals: decimals) }
    return values.count == 1 ? format(mean) : "\(format(mean)) (\(format(low))–\(format(high)))"
}

struct Totals {
    let rows: [TaskResult]
    func sum(_ key: KeyPath<TaskResult, Int>) -> Int { rows.reduce(0) { $0 + $1[keyPath: key] } }
    func sumD(_ key: KeyPath<TaskResult, Double>) -> Double { rows.reduce(0) { $0 + $1[keyPath: key] } }
    var correct: Int { rows.filter(\.correct).count }
    var fromStore: Int { rows.filter(\.fromStore).count }
    var peakMax: Int { rows.map(\.peakContext).max() ?? 0 }
    var exceeded: Int { rows.filter { $0.outcome == "context exceeded" }.count }
    var overLimit: Int { rows.filter { $0.peakContext > qualifyAbove }.count }
}

func reuseReport(_ passes: [[TaskResult]], model: Model, set: TaskSet, mode: String, storeBefore: Int?, storeAfter: Int?,
    kv: String?, started: Date, sourceCommit: String, serverFailed: Bool, facts: [(String, String)]) -> String
{
    let repeats = passes.count
    var lines = [
        "# Reuse benchmark: \(model.name), \(mode)",
        "",
        "| | |",
        "|---|---|",
        "| Run (UTC) | \(ISO8601DateFormatter().string(from: started)) |",
        "| Model | `\(model.file)` (\(model.repository) @ \(model.revision.prefix(12))) |",
        "| Mode | \(mode)\(storePath.map { ": `\($0)`" } ?? "") |",
        "| Repetitions | \(repeats)\(repeats > 1 ? ", each from the same starting store and a freshly started server (empty prompt cache)" : "") |",
        "| Store entries | \(storeBefore.map(String.init) ?? "—") before, \(storeAfter.map(String.init) ?? "—") after |",
        "| Context | \(reuseContext) tokens, KV cache \(kv ?? "f16"), 1 slot |",
        "| Task set | \(set.name): \(set.tasks.count) tasks (`\(set.path)`), authored against \(set.authoredAgainst) |",
        "| Source | `\(set.source)` at \(sourceCommit) |",
    ]
    lines += facts.map { "| \($0.0) | \($0.1) |" }
    lines += ["", serverFailed ? "**The server did not start** (see the server log): the model and context did not fit.\n" : ""]

    // Per pass totals.
    lines += [
        "## Totals per pass", "",
        "| Pass | Correct | From store | Context exceeded | Peak > \(grouped(Double(qualifyAbove), decimals: 0)) | Turns | Peak context (max) | Prompt sent | Prompt computed | Cache reuse | Generated | Wall s | Store: harness hits, model hits / puts, answers stored |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for (index, pass) in passes.enumerated() {
        let t = Totals(rows: pass)
        lines.append("| \(index + 1) | \(t.correct)/\(pass.count) | \(t.fromStore)/\(pass.count) | \(t.exceeded) | \(t.overLimit) | \(t.sum(\.turns)) | \(t.peakMax) | "
            + "\(t.sum(\.promptSent)) | \(t.sum(\.promptProcessed)) | \(reuseShare(t.sum(\.promptSent), t.sum(\.promptProcessed))) | "
            + "\(t.sum(\.generated)) | \(f1(t.sumD(\.wallSeconds))) | "
            + "\(t.sum(\.harnessHits)), \(t.sum(\.storeHits)) / \(t.sum(\.storePuts)), \(pass.filter(\.harnessPut).count) |")
    }
    if repeats > 1 {
        let all = passes.map { Totals(rows: $0) }
        lines.append("| **Mean (range)** | \(spread(all.map { Double($0.correct) }, decimals: 1)) | "
            + "\(spread(all.map { Double($0.fromStore) }, decimals: 1)) | \(spread(all.map { Double($0.exceeded) }, decimals: 1)) | "
            + "\(spread(all.map { Double($0.overLimit) }, decimals: 1)) | \(spread(all.map { Double($0.sum(\.turns)) })) | "
            + "\(spread(all.map { Double($0.peakMax) })) | \(spread(all.map { Double($0.sum(\.promptSent)) })) | "
            + "\(spread(all.map { Double($0.sum(\.promptProcessed)) })) | | \(spread(all.map { Double($0.sum(\.generated)) })) | "
            + "\(spread(all.map { $0.sumD(\.wallSeconds) }, decimals: 1)) | |")
    }

    // By overlap tag, when the task set has them.
    let tags = Array(Set(passes.flatMap { $0.map(\.tag) }.filter { !$0.isEmpty })).sorted()
    if !tags.isEmpty {
        lines += ["", "## By overlap tag (mean over passes)", "",
            "| Tag | Tasks | Correct | From store | Context exceeded | Prompt computed | Generated | Wall s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for tag in tags {
            let per = passes.map { Totals(rows: $0.filter { $0.tag == tag }) }
            let count = passes.first?.filter { $0.tag == tag }.count ?? 0
            lines.append("| \(tag) | \(count) | \(spread(per.map { Double($0.correct) }, decimals: 1)) | "
                + "\(spread(per.map { Double($0.fromStore) }, decimals: 1)) | \(spread(per.map { Double($0.exceeded) }, decimals: 1)) | "
                + "\(spread(per.map { Double($0.sum(\.promptProcessed)) })) | "
                + "\(spread(per.map { Double($0.sum(\.generated)) })) | \(spread(per.map { $0.sumD(\.wallSeconds) }, decimals: 1)) |")
        }
    }

    // Per task: mean (range) over passes.
    lines += ["", "## Per task\(repeats > 1 ? " (mean, with range over passes)" : "")", "",
        "| Task | Tag | Correct | From store | Context exceeded | Peak > \(grouped(Double(qualifyAbove), decimals: 0)) | Turns | Peak context | Prompt computed | Cache reuse | Generated | Wall s | Store: harness hits, model hits / puts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for task in set.tasks {
        let rows = passes.compactMap { $0.first { $0.id == task.id } }
        guard !rows.isEmpty else { continue }
        let sent = rows.reduce(0) { $0 + $1.promptSent }
        let computed = rows.reduce(0) { $0 + $1.promptProcessed }
        lines.append("| \(task.id) | \(task.tag) | \(rows.filter(\.correct).count)/\(rows.count) | \(rows.filter(\.fromStore).count)/\(rows.count) | "
            + "\(rows.filter { $0.outcome == "context exceeded" }.count)/\(rows.count) | \(rows.filter { $0.peakContext > qualifyAbove }.count)/\(rows.count) | "
            + "\(spread(rows.map { Double($0.turns) }, decimals: repeats > 1 ? 1 : 0)) | \(spread(rows.map { Double($0.peakContext) })) | "
            + "\(spread(rows.map { Double($0.promptProcessed) })) | \(reuseShare(sent, computed)) | \(spread(rows.map { Double($0.generated) })) | "
            + "\(spread(rows.map(\.wallSeconds), decimals: 1)) | "
            + "\(spread(rows.map { Double($0.harnessHits) }, decimals: repeats > 1 ? 1 : 0)), "
            + "\(spread(rows.map { Double($0.storeHits) }, decimals: repeats > 1 ? 1 : 0)) / "
            + "\(spread(rows.map { Double($0.storePuts) }, decimals: repeats > 1 ? 1 : 0)) |")
    }

    let wrong = passes.enumerated().flatMap { index, pass in
        pass.filter { !$0.correct && $0.outcome == "answered" }.map { "- pass \(index + 1), \($0.id): \($0.missing.joined(separator: "; "))" }
    }
    let unfinished = passes.enumerated().flatMap { index, pass in
        pass.filter { $0.outcome != "answered" }.map { "- pass \(index + 1), \($0.id): \($0.outcome)" }
    }
    if !wrong.isEmpty { lines += ["", "Missing from incorrect answers:"] + wrong }
    if !unfinished.isEmpty { lines += ["", "Not answered:"] + unfinished }
    lines += ["", "Prompt computed counts what the model had to process; cache reuse is the share of the prompt served from "
        + "llama-server's cache. From store: answered with no file tool call, from store entries alone. "
        + "Answers, and the entries each task was offered, are in the JSONL beside this report."]
    return lines.joined(separator: "\n") + "\n"
}

// MARK: Run

// SQLite's online backup: safe with WAL, unlike copying the file.
func backupStore(_ source: String, to destination: String) {
    try? FileManager.default.removeItem(atPath: destination)
    let result = queryInput(["sqlite3", source], input: ".backup \(sqlText(destination))\n")
    if !result.succeeded { fatal("cannot back up \(source) to \(destination): \(trimmed(result.output))") }
}

// Returns the report path and whether every task was answered correctly.
//
// P2, --repeat N: every pass starts from the same store state (a snapshot
// taken before the first) and from a freshly started server, so no pass
// inherits another's entries or prompt cache. Afterwards the store holds what
// the first pass left, as a single run would.
func runReuse(_ model: Model, ncmoe: [Int]?, kv: String?, base: String, started: Date, facts: [(String, String)]) -> (String, Bool) {
    guard FileManager.default.isExecutableFile(atPath: llamaServer) else {
        fatal("no \(llamaServer); build llama-server (SQLITE-STORE.md, \"Build\") or install it (00-setup-llama-cpp.swift).")
    }
    let set = loadTaskSet()
    let store = storePath.map { Store(path: $0) }
    store?.open()
    let storeBefore = store?.entryCount
    let mode = store == nil ? "baseline (no store)" : (storeBefore == 0 ? "store, cold" : "store, warm")
    let commit = trimmed(query(["git", "-C", set.source, "rev-parse", "--short", "HEAD"]).output)
    let dirty = !trimmed(query(["git", "-C", set.source, "status", "--porcelain", "--", "."]).output).isEmpty
    let sourceCommit = commit.isEmpty ? "not a git checkout" : commit + (dirty ? " (with uncommitted changes)" : "")
    print("    Tasks: \(set.tasks.count) from \(set.path)")
    print("    Source: \(set.source) at \(sourceCommit)")
    print("    Mode: \(mode)\(store.map { " (\($0.path), \(storeBefore ?? 0) entries)" } ?? "")"
        + (reuseRepeats > 1 ? ", \(reuseRepeats) passes" : ""))

    let scratch = "\(base)-scratch"
    try? FileManager.default.createDirectory(atPath: scratch, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(atPath: scratch) }
    let runId = "\((base as NSString).lastPathComponent)"
    let snapshot = "\(scratch)/start.sqlite"
    if let store, reuseRepeats > 1 { backupStore(store.path, to: snapshot) }

    var passes: [[TaskResult]] = []
    var serverFailed = false
    for pass in 1...reuseRepeats {
        // A single pass works on the store itself; repeated passes on copies of the start.
        var passStore = store
        if store != nil, reuseRepeats > 1 {
            let copy = "\(scratch)/pass-\(pass).sqlite"
            backupStore(snapshot, to: copy)
            passStore = Store(path: copy)
        }
        let context = ReuseContext(store: passStore, root: set.source, run: "\(runId)-pass\(pass)",
            model: "\(model.name)@\(model.sha256.prefix(12))")
        if reuseRepeats > 1 { print("    Pass \(pass) of \(reuseRepeats)") }
        var results: [TaskResult] = []
        let log = "\(base)-server\(reuseRepeats > 1 ? "-pass\(pass)" : "").log"
        guard let server = startServer(model, ncmoe: ncmoe, kv: kv, log: log) else {
            serverFailed = true
            print("    The server did not start: see \(log) (out of memory at this context?)")
            passes.append(set.tasks.map {
                var r = TaskResult(id: $0.id)
                r.tag = $0.tag
                r.role = $0.role
                r.repeatIndex = pass
                r.outcome = "server did not start"
                return r
            })
            break
        }
        for task in set.tasks {
            print("    [\(task.id)] ", terminator: "")
            var result = runTask(task, context, server: server, scratch: scratch)
            result.repeatIndex = pass
            print("\(result.outcome), \(result.correct ? "correct" : "incorrect"), peak \(result.peakContext) tokens, "
                + (result.refusedRequest > 0 ? "refused request \(result.refusedRequest) tokens, " : "")
                + "\(result.promptProcessed) computed, \(String(format: "%.0f", result.wallSeconds)) s"
                + (passStore == nil ? "" : ", store: \(result.harnessHits) harness hits, \(result.storeHits) model hits / "
                    + "\(result.storePuts) puts, \(result.fromStore ? "from store" : "read files"), "
                    + "answer \(result.harnessPut ? "stored" : "not stored")"))
            results.append(result)
            if result.outcome == "server stopped" { break }
        }
        stopServer(server)
        passes.append(results)
        if results.last?.outcome == "server stopped" { break }
    }
    // The store ends as the first pass left it.
    if let store, reuseRepeats > 1, FileManager.default.fileExists(atPath: "\(scratch)/pass-1.sqlite") {
        backupStore("\(scratch)/pass-1.sqlite", to: store.path)
    }
    let jsonl = passes.flatMap { $0 }.map { jsonString($0.asJSON) }.joined(separator: "\n") + "\n"
    try? jsonl.write(toFile: "\(base).jsonl", atomically: true, encoding: .utf8)
    let markdown = reuseReport(passes, model: model, set: set, mode: mode, storeBefore: storeBefore, storeAfter: store?.entryCount,
        kv: kv, started: started, sourceCommit: sourceCommit, serverFailed: serverFailed, facts: facts)
    try? markdown.write(toFile: "\(base).md", atomically: true, encoding: .utf8)
    return ("\(base).md", !serverFailed && passes.allSatisfy { $0.allSatisfy(\.correct) })
}

// MARK: - Commands

func printUsage() {
    print("""
    Usage:
      bench/bench.swift list [<selector>] [--fit vram|offload]
      bench/bench.swift capabilities [<selector>]
      bench/bench.swift fetch <selector> [--fit vram|offload] [--dry-run]
      bench/bench.swift run <selector> <suite> [--build <dir> | --accelerator <name>] [--fit vram|offload]
                                   [--ncmoe <list>] [--threads <list>] [--reps <n>] [--dry-run]

    <selector>  A model name, a purpose (\(Purpose.allCases.map(\.rawValue).joined(separator: ", "))), or all.
                A purpose selects every model declaring it, primary-purpose models first.
                graphics means reading images; the projector is fetched with the model.
    --fit       vram: fits the dGPU whole. offload: expert weights in system RAM.
    --build     A build directory's bin (default \(defaultBuildDirectory) when nothing is installed):
                its llama-bench and llama-server run directly, on that build's backend.
    --accelerator  With an install by 00-setup-llama-cpp.swift: where llama.cpp runs, through llama-on
                (default nvidia). Installed:
                \(installedAcceleratorNames().joined(separator: ", ")). Off the dGPU (intel, cpu), the model is whole
                on the device unless --ncmoe says otherwise, the dGPU is never queried, and the offload suite does not apply.

    list      Declared models by purpose, whether each is present, and the suites.
    capabilities  Each model's declared capabilities (input, tool calls, thinking switch, native context, the
              knowledge cutoff its template states), checked against its fetched files; prints MODELS.md's
              table. fetch runs the same check.
    fetch     Download the selected models, as you, to \(userModelDirectory),
              pinned by revision and SHA-256. Models already in \(systemModelDirectory) are used there.
    run       Run the suite on each selected model that is present, and write one report per model to
              \(resultsDirectory)/. Refuses while a llama-server service is running (each stops by itself
              2 minutes after its last connection) or, on the dGPU, while anything holds it. A service started
              during the run makes that model's result invalid, and the report says so.

    No command needs root, and none uses sudo.
      --ncmoe <list>    Expert layers kept in RAM, e.g. 26 or 36,28,24 (offload models).
      --threads <list>  CPU threads, e.g. 8 or 8,16.
      --kv <type>       KV cache type: f16 (default), q8_0 or q4_0. q8_0 halves the cache for long contexts.
      --ub <n>          Micro-batch size (llama.cpp -ub; default 512). On the iGPU, 128 keeps each compute
                        submission inside i915's preempt timeout; at 512 the dense models hang the GPU and abort
                        (igpu-workloads.md).

    The reuse suite (real tasks through tools, not llama-bench):
      --store <file>    Turn the reuse store on: an SQLite file you own. Run twice for cold, then warm.
      --ctx <n>         The server's context size (default 32768).
      --tasks <file>    Task set (default reuse-tasks-c++.json beside this script).
      --source <dir>    The tree the tasks are about (default: the first of the task set's own "source" paths that exists).
      --role <r>        Only the measured or only the build tasks of the task set (default: all).
      --qualify-above <n>  Count tasks whose peak context exceeds n (default 32768; a test threshold, not a hardware limit).
      --repeat <n>      Passes over the task set (default 1). Each starts from the same store state and a fresh
                        server; the report gives the mean and range. The store ends as the first pass left it.
      --reps <n>        Repetitions per test (default 3).

    The state suite (the KV-cache state store against an out-of-memory error; needs a llama-bench
    with --store, from this repository): run 1 (no store; must run out of memory, or it halts),
    the fill (-ub 256, saves the depth), runs 2-4 (store and CUDA graphs on/off), at -p 0 -n 128.
      --store <file>    The SQLite file for the depth state (required). Several GB per state.
      --depth <n>       The depth in tokens (default 65280).
    """)
}

var fitFilter: Fit?

// A model name, a purpose, or `all`; then the --fit filter.
func select(_ selector: String?) -> [Model] {
    let chosen: [Model]
    switch selector {
    case nil, "all"?:
        chosen = models
    case let name? where Purpose(rawValue: name) != nil:
        let purpose = Purpose(rawValue: name)!
        chosen = models.filter { $0.purposes.contains(purpose) }
            .sorted { ($0.primary == purpose ? 0 : 1) < ($1.primary == purpose ? 0 : 1) }
    case let name?:
        guard let found = models.first(where: { $0.name == name }) else {
            fatal("unknown model or purpose '\(name)'; models: \(models.map(\.name).joined(separator: ", "));",
                "purposes: \(Purpose.allCases.map(\.rawValue).joined(separator: ", ")), or all.")
        }
        chosen = [found]
    }
    let filtered = chosen.filter { fitFilter == nil || $0.fit == fitFilter }
    if filtered.isEmpty { fatal("no declared model matches '\(selector ?? "all")'\(fitFilter.map { " with --fit \($0.rawValue)" } ?? "").") }
    return filtered
}

func list(_ selected: [Model]) {
    print("Models (\(userModelDirectory), then \(systemModelDirectory)):")
    for purpose in Purpose.allCases {
        let group = selected.filter { $0.primary == purpose }
        guard !group.isEmpty else { continue }
        print("  \(purpose.rawValue):")
        for model in group {
            let present = model.present
            let size = String(format: "%.1f GB", Double(model.downloadBytes) / 1e9)
            let also = model.purposes.dropFirst().map(\.rawValue)
            print("    \(model.name.padding(toLength: 18, withPad: " ", startingAt: 0))"
                + "\(size.padding(toLength: 9, withPad: " ", startingAt: 0))"
                + "\(model.fit.rawValue.padding(toLength: 8, withPad: " ", startingAt: 0))"
                + "\((present ? "present" : "-").padding(toLength: 8, withPad: " ", startingAt: 0))"
                + "\(model.note)\(model.projector == nil ? "" : "; image projector")"
                + "\(also.isEmpty ? "" : " (also: \(also.joined(separator: ", ")))")")
        }
    }
    print("Suites:")
    for suite in Suite.allCases { print("    \(suite.rawValue.padding(toLength: 9, withPad: " ", startingAt: 0)) \(suite.summary)") }
    print("Results: \(resultsDirectory)")
}

// Downloads beside the destination, checks the SHA-256, then moves into place.
// As you, into your store only.
func fetchFile(url: String, path: String, sha256: String, bytes: Int64) {
    if fileSize(path) == bytes {
        print("    Present: \(path)")
        return
    }
    let destination = "\(userModelDirectory)/\((path as NSString).lastPathComponent)"
    let partial = "\(userModelDirectory)/.\((destination as NSString).lastPathComponent).part"
    print("    >>> mkdir -p \(userModelDirectory)")
    if !dryRun {
        do { try FileManager.default.createDirectory(atPath: userModelDirectory, withIntermediateDirectories: true) } catch {
            fatal("cannot create \(userModelDirectory): \(error)")
        }
    }
    // -C - resumes an interrupted download.
    mutate(["curl", "-fL", "--retry", "3", "-C", "-", "-o", partial, url])
    guard !dryRun else {
        print("    >>> sha256sum \(partial); mv \(partial) \(destination)")
        return
    }
    print("    Checking SHA-256...")
    let digest = query(["sha256sum", partial]).output.split(separator: " ").first.map(String.init) ?? ""
    guard digest == sha256 else {
        try? FileManager.default.removeItem(atPath: partial)
        fatal("\(url): SHA-256 \(digest) does not match the pinned \(sha256). Removed it.")
    }
    print("    SHA-256 matches the pin.")
    do { try FileManager.default.moveItem(atPath: partial, toPath: destination) } catch {
        fatal("cannot move \(partial) to \(destination): \(error)")
    }
    print("    Stored: \(destination)")
}

// MARK: - State suite (the KV-cache state store against an out-of-memory error)
//
// The sequence of llama-bench-oom-reuse-sqlite.md, at -p 0 -n 128 -d <depth>:
//   run 1   no store, CUDA graphs on: must run out of memory, or the test halts
//   fill    run 1 plus -ub 256 --store: computes the depth at a smaller
//           micro-batch, which fits, and saves it
//   run 2   store, CUDA graphs on
//   run 3   no store, CUDA graphs off (the control)
//   run 4   store, CUDA graphs off
// The store key leaves the micro-batch out, so runs 2 and 4 restore at the
// settings that failed. Needs a llama-bench with --store (this repository).

var stateDepth = 65_280

struct StateRun {
    let name: String
    let store: Bool
    let graphs: Bool
    let fill: Bool
}

let stateRuns = [
    StateRun(name: "run 1", store: false, graphs: true, fill: false),
    StateRun(name: "fill", store: true, graphs: true, fill: true),
    StateRun(name: "run 2", store: true, graphs: true, fill: false),
    StateRun(name: "run 3", store: false, graphs: false, fill: false),
    StateRun(name: "run 4", store: true, graphs: false, fill: false),
]

// stdout and stderr together into a file; the exit status back.
func spawnLogged(_ argv: [String], to file: String) -> Int32 {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = argv
    _ = FileManager.default.createFile(atPath: file, contents: nil)
    guard let handle = FileHandle(forWritingAtPath: file) else { fatal("cannot write \(file)") }
    process.standardOutput = handle
    process.standardError = handle
    do { try process.run() } catch { return 127 }
    process.waitUntilExit()
    return process.terminationReason == .uncaughtSignal ? 128 + process.terminationStatus : process.terminationStatus
}

func runStateStore(_ model: Model, kv: String?, reps: Int, base: String, started: Date, facts: [(String, String)]) -> (String, Bool) {
    guard let store = storePath else { fatal("the state suite needs --store <sqlite>: the file that holds the depth state.") }
    let help = query(["sh", "-c", "exec \"$@\" --help 2>&1", "sh"] + onAccelerator("llama-bench")).output
    guard help.contains("--store") else {
        fatal("this llama-bench has no --store: build this repository (SQLITE-STORE.md), and pass --build <dir>.")
    }
    var common = onAccelerator("llama-bench") + ["-m", model.path, "-fa", "on", "-r", "\(reps)", "-o", "md",
        "-p", "0", "-n", "128", "-d", "\(stateDepth)"]
    if let kv { common += ["-ctk", kv, "-ctv", kv] }

    struct Outcome { let run: StateRun; let argv: [String]; let status: Int32; let rows: [String]; let oom: [String]; let storeLines: [String] }
    var outcomes: [Outcome] = []
    var halted: String?
    for run in stateRuns {
        var argv = run.graphs ? common : ["env", "GGML_CUDA_DISABLE_GRAPHS=1"] + common
        if run.fill { argv += ["-ub", "256"] }
        if run.store { argv += ["--store", store] }
        let log = "\(base)-\(run.name.replacingOccurrences(of: " ", with: "")).log"
        print("    \(run.name) (\(ISO8601DateFormatter().string(from: Date())))")
        print("    >>> \(argv.joined(separator: " "))")
        let status = spawnLogged(argv, to: log)
        let lines = ((try? String(contentsOfFile: log, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init)
        let rows = lines.filter { $0.hasPrefix("| ") && !$0.hasPrefix("| model") && !$0.hasPrefix("| -") }
        let oom = lines.filter { $0.contains("out of memory") || $0.contains("cuMemCreate") || $0.contains("cudaGraphInstantiate") }
        let storeLines = lines.filter { $0.contains("store:") }
        for line in rows + oom.prefix(3) + storeLines { print("      \(line)") }
        print("      exit \(status)")
        outcomes.append(Outcome(run: run, argv: argv, status: status, rows: rows, oom: oom, storeLines: storeLines))
        if status == 130 || status == 143 { halted = "interrupted"; break }
        if run.name == "run 1", oom.isEmpty {
            halted = "run 1 did not run out of memory, so this model, depth and GPU do not reproduce the error the store is tested against"
            break
        }
    }

    var md = "# State store: \(model.name), depth \(stateDepth)\n\n"
    md += "Started \(ISO8601DateFormatter().string(from: started)). Store: `\(store)`.\n\n"
    md += "| | |\n|---|---|\n" + facts.map { "| \($0.0) | \($0.1) |" }.joined(separator: "\n") + "\n\n"
    md += "| Run | Store | CUDA graphs | Result |\n|---|---|---|---|\n"
    for o in outcomes {
        let result: String
        if o.status == 0, !o.rows.isEmpty {
            result = "finished: " + o.rows.compactMap { $0.split(separator: "|").last.map { $0.trimmingCharacters(in: .whitespaces) } }.joined(separator: "; ") + " t/s"
        } else if !o.oom.isEmpty {
            result = "out of memory" + (o.oom.contains { $0.contains("cudaGraphInstantiate") } ? " (building a CUDA graph)" : "")
        } else {
            result = "exit \(o.status)"
        }
        md += "| \(o.run.name) | \(o.run.store ? (o.run.fill ? "fill (-ub 256)" : "yes") : "no") | \(o.run.graphs ? "on" : "off") | \(result) |\n"
    }
    if let halted { md += "\n**Halted:** \(halted).\n" }
    md += "\n## Commands\n\n" + outcomes.map { "- \($0.run.name): `\($0.argv.joined(separator: " "))`" }.joined(separator: "\n") + "\n"
    md += "\n## Store lines\n\n" + outcomes.flatMap { o in o.storeLines.map { "- \(o.run.name): \($0)" } }.joined(separator: "\n") + "\n"
    let path = "\(base).md"
    if (try? md.write(toFile: path, atomically: true, encoding: .utf8)) == nil { fatal("cannot write \(path)") }
    return (path, halted == nil)
}

func fetch(_ model: Model) {
    fetchFile(url: model.url, path: model.path, sha256: model.sha256, bytes: model.bytes)
    if let projector = model.projector, let path = model.projectorPath, let url = model.projectorURL {
        fetchFile(url: url, path: path, sha256: projector.sha256, bytes: projector.bytes)
    }
}

func run(_ selected: [Model], _ suite: Suite, ncmoe: [Int]?, threads: [Int]?, kv: String?, reps: Int) {
    guard buildDirectory != nil || FileManager.default.isExecutableFile(atPath: llamaOn) else {
        fatal("no \(llamaOn); pass --build <dir>, or install with 00-setup-llama-cpp.swift.")
    }
    // The offload suite applies only to offload models; a group skips the rest.
    let applicable = suite == .offload ? selected.filter { $0.offload != nil } : selected
    if applicable.isEmpty { fatal("none of the selected models is an offload model; the offload suite needs one.") }
    let present = applicable.filter { fileSize($0.path) == $0.bytes }
    for missing in applicable where fileSize(missing.path) != missing.bytes {
        print("Skipping \(missing.name): not fetched (bench/bench.swift fetch \(missing.name)).")
    }
    if present.isEmpty { fatal("none of the selected models is present; fetch one first.") }

    print("Benchmark: \(suite.rawValue) (\(suite.summary)) on \(present.map(\.name).joined(separator: ", ")), "
        + "accelerator \(accelerator.name) (\(accelerator.summary))")
    // A running server holds its device's memory and competes for the CPU.
    // Each stops by itself after its idle time; nothing here needs root.
    let servers = serverUnits()
    let running = servers.filter { trimmed(query(["systemctl", "is-active", $0]).output) == "active" }
    if !running.isEmpty {
        fatal("\(running.joined(separator: ", ")) running: it holds its device's memory and would compete with the run.",
            "It stops by itself 2 minutes after its last connection; run again once it has.")
    }
    if accelerator.usesDGPU {
        let holders = gpuComputePids()
        if !holders.isEmpty {
            fatal("the GPU is in use by: \(holders.joined(separator: "; ")). Stop it and re-run.")
        }
    }

    if !dryRun { try? FileManager.default.createDirectory(atPath: resultsDirectory, withIntermediateDirectories: true) }
    let facts = dryRun ? [] : hostFacts()
    var written: [String] = []
    var failed: [String] = []
    // Ctrl-C goes to llama-bench (handlers reset on exec); this process
    // survives it to write the report of what was measured.
    signal(SIGINT) { _ in }
    signal(SIGTERM) { _ in }
    for model in present {
        let argv = benchArguments(model, suite, ncmoe: ncmoe, threads: threads, kv: kv, reps: reps)
        let started = Date()
        let stamp = ISO8601DateFormatter().string(from: started).replacingOccurrences(of: ":", with: "")
        let base = "\(resultsDirectory)/\(stamp)-\(model.name)-\(suite.rawValue)\(kv.map { "-kv-\($0)" } ?? "")"
            + (accelerator.name == "nvidia" ? "" : "-\(accelerator.name)") + (ubatch.map { "-ub\($0)" } ?? "")
        let serversBefore = invocationIDs(servers)
        // A server started during the run (a client connected) shared the
        // device: the measurement is not this model's alone.
        func serverStarted() -> String? {
            let after = invocationIDs(servers)
            let started = servers.filter { after[$0] != serversBefore[$0] }
            return started.isEmpty ? nil : started.joined(separator: ", ")
        }
        print("\n== \(model.name) (\(model.purposes.map(\.rawValue).joined(separator: ", ")); \(model.fit.rawValue))")
        if suite == .reuse {
            let mode = storePath == nil ? "baseline" : "store"
            let reuseFiles = "\(base)-\(mode)-ctx\(reuseContext)"
            guard !dryRun else {
                let set = loadTaskSet()
                print("    Would run \(set.tasks.count) tasks from \(set.path) on \(set.source), context \(reuseContext), "
                    + (storePath.map { "store \($0)" } ?? "no store"))
                continue
            }
            let sampler = accelerator.usesDGPU ? startGpuSampler("\(reuseFiles)-gpu.csv") : nil
            let (path, allCorrect) = runReuse(model, ncmoe: ncmoe, kv: kv, base: reuseFiles, started: started, facts: facts)
            stopGpuSampler(sampler)
            if accelerator.usesDGPU { appendGpuSection(toReport: path, csv: "\(reuseFiles)-gpu.csv") }
            written.append(path)
            if !allCorrect { failed.append("\(model.name) (not every task answered correctly; see the report)") }
            if let started = serverStarted() { failed.append("\(model.name) (invalid: \(started) started during the run)") }
            continue
        }
        if suite == .state {
            guard !dryRun else {
                print("    Would run run 1, fill, runs 2-4 at depth \(stateDepth), store \(storePath ?? "(none: --store needed)")")
                continue
            }
            let (path, completed) = runStateStore(model, kv: kv, reps: reps, base: "\(base)-d\(stateDepth)", started: started, facts: facts)
            written.append(path)
            if !completed { failed.append("\(model.name) (halted; see the report)") }
            if let started = serverStarted() { failed.append("\(model.name) (invalid: \(started) started during the run)") }
            continue
        }
        print("    >>> \(argv.joined(separator: " "))")
        guard !dryRun else { continue }
        let sampler = accelerator.usesDGPU ? startGpuSampler("\(base)-gpu.csv") : nil
        let status = spawn(argv, stdoutTo: "\(base).jsonl")
        stopGpuSampler(sampler)
        let interloper = serverStarted()
        var markdown = report(jsonl: "\(base).jsonl", model: model, suite: suite, argv: argv, started: started, facts: facts)
        markdown += accelerator.usesDGPU ? gpuSection("\(base)-gpu.csv")
            : "\n## GPU during the run\n\nNot sampled: the run does not use the dGPU, and nvidia-smi would wake it.\n"
        if let interloper {
            markdown += "\n## Invalid\n\n\(interloper) started during the run and shared the machine; run again.\n"
            failed.append("\(model.name) (invalid: \(interloper) started during the run)")
        }
        if (try? markdown.write(toFile: "\(base).md", atomically: true, encoding: .utf8)) != nil { written.append("\(base).md") }
        if status != 0 {
            failed.append("\(model.name) (exit \(status))")
            // Interrupted: stop the group, not just this model.
            if status == 130 || status == 143 { break }
        }
    }
    signal(SIGINT, SIG_DFL)
    signal(SIGTERM, SIG_DFL)
    guard !dryRun else {
        print("Dry run complete — nothing changed.")
        return
    }
    print("\nResults:")
    for path in written { print("    \(path)") }
    if !failed.isEmpty {
        fatal(suite == .reuse ? "reuse: \(failed.joined(separator: "; "))"
            : "llama-bench failed for: \(failed.joined(separator: "; ")); those reports are partial.")
    }
}

// MARK: - Arguments

func intList(_ text: String?, _ flag: String) -> [Int] {
    guard let text, case let values = text.split(separator: ",").compactMap({ Int($0) }),
        !values.isEmpty, values.count == text.split(separator: ",").count
    else { fatal("\(flag) takes a comma-separated list of integers.") }
    return values
}

var positional: [String] = []
var ncmoe: [Int]?
var threads: [Int]?
var reps = 3
var kv: String?
var arguments = CommandLine.arguments.dropFirst().makeIterator()
while let arg = arguments.next() {
    switch arg {
    case "--dry-run": dryRun = true
    case "--accelerator":
        guard let value = arguments.next(), !value.hasPrefix("-") else { fatal("--accelerator takes an accelerator name.") }
        acceleratorName = value
        acceleratorGiven = true
    case "--build":
        guard let value = arguments.next(), !value.hasPrefix("-") else { fatal("--build takes a build directory (its bin).") }
        buildDirectory = URL(fileURLWithPath: (value as NSString).expandingTildeInPath).standardizedFileURL.path
    case "--ncmoe": ncmoe = intList(arguments.next(), "--ncmoe")
    case "--threads": threads = intList(arguments.next(), "--threads")
    case "--fit":
        guard let value = arguments.next(), let fit = Fit(rawValue: value) else { fatal("--fit takes vram or offload.") }
        fitFilter = fit
    case "--ub":
        guard let n = arguments.next().flatMap({ Int($0) }), (1...8192).contains(n) else { fatal("--ub takes a micro-batch size, 1 to 8192.") }
        ubatch = n
    case "--kv":
        guard let value = arguments.next(), ["f16", "q8_0", "q4_0"].contains(value) else { fatal("--kv takes f16, q8_0 or q4_0.") }
        kv = value == "f16" ? nil : value
    case "--store":
        guard let value = arguments.next(), !value.hasPrefix("-") else { fatal("--store takes a path to an SQLite file.") }
        storePath = URL(fileURLWithPath: (value as NSString).expandingTildeInPath).standardizedFileURL.path
    case "--depth":
        guard let n = arguments.next().flatMap({ Int($0) }), n > 0 else { fatal("--depth takes a token count.") }
        stateDepth = n
    case "--ctx":
        guard let n = arguments.next().flatMap({ Int($0) }), n >= 2048 else { fatal("--ctx takes a context size of at least 2048.") }
        reuseContext = n
    case "--tasks":
        guard let value = arguments.next() else { fatal("--tasks takes a path to a task-set JSON file.") }
        tasksPath = URL(fileURLWithPath: (value as NSString).expandingTildeInPath).standardizedFileURL.path
    case "--source":
        guard let value = arguments.next() else { fatal("--source takes a directory.") }
        sourcePath = URL(fileURLWithPath: (value as NSString).expandingTildeInPath).standardizedFileURL.path
    case "--role":
        guard let value = arguments.next(), ["measured", "build"].contains(value) else { fatal("--role takes measured or build.") }
        roleFilter = value
    case "--qualify-above":
        guard let n = arguments.next().flatMap({ Int($0) }), n > 0 else { fatal("--qualify-above takes a token count.") }
        qualifyAbove = n
    case "--repeat":
        guard let n = arguments.next().flatMap({ Int($0) }), (1...20).contains(n) else { fatal("--repeat takes 1 to 20.") }
        reuseRepeats = n
    case "--reps":
        guard let n = arguments.next().flatMap({ Int($0) }), n > 0 else { fatal("--reps takes a positive integer.") }
        reps = n
    case "-h", "--help":
        printUsage()
        exit(0)
    default:
        if arg.hasPrefix("-") {
            printUsage()
            fatal("unknown option '\(arg)'.")
        }
        positional.append(arg)
    }
}

// One authority: the invoking user's. Under sudo, files would land root-owned
// in the wrong home, and swift itself is usually not on root's PATH.
if geteuid() == 0 {
    fatal("do not run this under sudo or as root; it needs no privilege.",
        "Run it as yourself.")
}

// Build mode with --build, or when nothing is installed; otherwise the install.
if buildDirectory == nil, !FileManager.default.isExecutableFile(atPath: llamaOn) {
    buildDirectory = defaultBuildDirectory
}
if let directory = buildDirectory {
    if acceleratorGiven { fatal("--accelerator selects an installed accelerator; with --build, the build's backend is used.") }
    if ["run"].contains(positional.first) { accelerator = buildAccelerator(directory) }
} else {
    guard let installed = installedAccelerator(acceleratorName) else {
        fatal("no accelerator '\(acceleratorName)' in \(acceleratorDirectory); installed: "
            + "\(installedAcceleratorNames().joined(separator: ", ")) (00-setup-llama-cpp.swift install --vulkan adds intel and cpu).")
    }
    accelerator = installed
}

switch positional.first {
case "list":
    list(select(positional.dropFirst().first))
case "fetch":
    guard let selector = positional.dropFirst().first else { fatal("fetch needs a model, a purpose, or all.") }
    let chosen = select(selector)
    let total = chosen.filter { !$0.present }.reduce(Int64(0)) { $0 + $1.downloadBytes }
    print("Fetching \(chosen.map(\.name).joined(separator: ", ")) (\(String(format: "%.1f", Double(total) / 1e9)) GB to download)")
    for model in chosen {
        print("\n== \(model.name)")
        fetch(model)
    }
    if !dryRun {
        print("\nCapabilities, checked against the files:")
        if !checkCapabilities(chosen) { fatal("a declaration does not match its files; correct declaredModels.") }
    }
case "capabilities":
    print("Capabilities: declared in declaredModels, checked against each fetched model's files.")
    if !checkCapabilities(select(positional.dropFirst().first)) { fatal("a declaration does not match its files.") }
case "run":
    guard let selector = positional.dropFirst().first else { fatal("run needs a model, a purpose, or all.") }
    guard let name = positional.dropFirst(2).first, let suite = Suite(rawValue: name) else {
        fatal("run needs a suite: \(Suite.allCases.map(\.rawValue).joined(separator: ", ")).")
    }
    run(select(selector), suite, ncmoe: ncmoe, threads: threads, kv: kv, reps: reps)
default:
    printUsage()
    exit(1)
}
