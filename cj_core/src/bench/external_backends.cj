package semanticvg_cj.bench

import std.process.*
import std.core.*

// 仓颉 benchmark 只依赖此协议；数据库 SDK 和版本差异封装在独立适配器进程中。
public interface ExternalVectorBackend {
    func backendName(): String
    func run(datasetId: String, scale: String): String
}

public class ProcessExternalVectorBackend <: ExternalVectorBackend {
    private let name: String
    private let pythonCommand: String

    public init(name: String, pythonCommand!: String = "python") {
        if (name != "mock" && name != "milvus" && name != "qdrant" && name != "chroma") {
            throw IllegalArgumentException("未知外部数据库 backend: ${name}")
        }
        this.name = name
        this.pythonCommand = pythonCommand
    }

    public func backendName(): String { this.name }

    public func run(datasetId: String, scale: String): String {
        let args = [
            "../tools/external_db_benchmark.py",
            "--backend", this.name,
            "--dataset", datasetId,
            "--scale", scale
        ]
        let (exitCode, stdoutBytes, stderrBytes) = executeWithOutput(this.pythonCommand, args)
        let stdout = String.fromUtf8(stdoutBytes)
        let stderr = String.fromUtf8(stderrBytes)
        if (exitCode != 0) {
            throw Exception("外部数据库 benchmark 失败: backend=${this.name}, exit=${exitCode}\n${stdout}\n${stderr}")
        }
        for (line in stdout.split("\n")) {
            if (line.startsWith("EXTERNAL_DB_RESULT|")) {
                return line[19..line.size]
            }
        }
        throw Exception("外部数据库 benchmark 未返回 EXTERNAL_DB_RESULT: ${stdout}")
    }
}

public func runExternalDbPreset(backendName: String, datasetId: String, scale: String): Unit {
    if (findSpec(datasetId).isNone()) {
        throw IllegalArgumentException("dbbench 数据集必须为 1..6")
    }
    if (scale != "smoke" && scale != "partial" && scale != "full") {
        throw IllegalArgumentException("dbbench scale 必须为 smoke/partial/full")
    }
    let backend: ExternalVectorBackend = ProcessExternalVectorBackend(backendName)
    println("启动外部数据库 benchmark: backend=${backend.backendName()}, dataset=${datasetId}, scale=${scale}")
    let resultJson = backend.run(datasetId, scale)
    println("DB_RESULT_JSON|${resultJson}")
}
