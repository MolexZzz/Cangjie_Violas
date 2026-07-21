package semanticvg_cj.bench

import std.collection.*
import std.core.*

public class DatasetSpec {
    public let id: String
    public let label: String
    public let txtPath: String
    public let vectorType: String
    public let isText: Bool

    public init(id: String, label: String, txtPath: String, vectorType: String, isText: Bool) {
        this.id = id
        this.label = label
        this.txtPath = txtPath
        this.vectorType = vectorType
        this.isText = isText
    }
}

public func allDatasetSpecs(): Array<DatasetSpec> {
    let base = "../dataset/precomputed/"
    let list = ArrayList<DatasetSpec>()
    list.add(DatasetSpec("1", "20 Newsgroups", base + "news20_precomputed.txt", "text", true))
    list.add(DatasetSpec("2", "OHSUMED", base + "ohsumed_precomputed.txt", "text", true))
    list.add(DatasetSpec("3", "Yahoo Answers", base + "yahoo_precomputed.txt", "text", true))
    list.add(DatasetSpec("4", "Caltech-101", base + "caltech_precomputed.txt", "image", false))
    list.add(DatasetSpec("5", "CUB-200", base + "cub_precomputed.txt", "image", false))
    list.add(DatasetSpec("6", "COCO (scope由manifest确认)", base + "coco_precomputed.txt", "image", false))
    return list.toArray()
}

public func findSpec(key: String): Option<DatasetSpec> {
    for (s in allDatasetSpecs()) {
        if (s.id == key || s.label == key) { return Some(s) }
    }
    None
}

public func runSpecs(specs: Array<DatasetSpec>, scale: EvalScale): Unit {
    for (s in specs) {
        println("\n" + "#" * 72)
        println(">>> 数据集: ${s.label}")
        println("#" * 72)
        let dsScale = EvalScale(
            scale.nFolds, scale.maxFolders, scale.maxQueries, scale.maxVectorsPerFolder,
            scale.topK, s.vectorType, scale.repMethod
        )
        runFullBenchmark(s.txtPath, s.label, dsScale)
    }
}

public func printDatasetMenu(): Unit {
    println("\n数据集:")
    for (s in allDatasetSpecs()) {
        println("  ${s.id} = ${s.label}  (${s.txtPath})")
    }
    println("  t = 文本三个 (1+2+3)")
    println("  v = 图像三个 (4+5+6)")
    println("  a = 全部六个")
}

public func parseSelection(input: String): Array<DatasetSpec> {
    let trimmed = input.trimAscii()
    let all = allDatasetSpecs()
    if (trimmed.isEmpty() || trimmed == "a") { return all }
    if (trimmed == "t") {
        let t = ArrayList<DatasetSpec>()
        t.add(all[0]); t.add(all[1]); t.add(all[2])
        return t.toArray()
    }
    if (trimmed == "v") {
        let v = ArrayList<DatasetSpec>()
        v.add(all[3]); v.add(all[4]); v.add(all[5])
        return v.toArray()
    }
    let result = ArrayList<DatasetSpec>()
    let parts = trimmed.split(",")
    for (part in parts) {
        let p = part.trimAscii()
        match (findSpec(p)) {
            case Some(s) => result.add(s)
            case None => println("  忽略未知项: ${p}")
        }
    }
    if (result.isEmpty()) { return all }
    return result.toArray()
}

public func askEvalScale(): EvalScale {
    println("\n=== 仓颉评测规模 ===")
    println("  1 = 部分（每类最多 120 条向量；每种模式最多 200 条查询，推荐）")
    println("  2 = 全量（全部向量与全部查询，大文件易 OOM）")
    println("  3 = 小样本（每类最多 20 条向量；每种模式最多 20 条查询）")
    print("请选择 [1/2/3，默认 1]: ")
    let line = readln()
    if (line == "2") {
        return EvalScale.full("text")
    }
    if (line == "3") {
        return EvalScale.smoke("text")
    }
    return EvalScale.partial("text")
}

public func askDatasetSelection(): Array<DatasetSpec> {
    printDatasetMenu()
    print("\n请选择数据集（编号逗号分隔，或 t/v/a）: ")
    let line = readln()
    return parseSelection(line)
}

public func interactiveBenchMain(): Unit {
    println("=" * 72)
    println("semanticvg 仓颉评测（读取 dataset/precomputed/*.txt）")
    println("工作目录应为 cj_core/，请先完成 Python 预处理导出。")
    println("=" * 72)

    var scale = askEvalScale()
    let specs = askDatasetSelection()

    println("\n将评测 ${specs.size} 个数据集")
    for (s in specs) {
        let dsScale = EvalScale(
            scale.nFolds, scale.maxFolders, scale.maxQueries, scale.maxVectorsPerFolder,
            scale.topK, s.vectorType, scale.repMethod
        )
        runFullBenchmark(s.txtPath, s.label, dsScale)
    }
    println("\nAll benchmarks finished.")
}

// 单行可自动化入口，例如：echo "bench smoke 1" | cjpm run
public func runBenchPreset(scaleName: String, selection: String): Unit {
    let specs = parseSelection(selection)
    if (scaleName != "smoke" && scaleName != "partial" && scaleName != "full") {
        throw IllegalArgumentException("未知评测规模: ${scaleName}，可选 smoke/partial/full")
    }
    for (s in specs) {
        let scale = if (scaleName == "smoke") { EvalScale.smoke(s.vectorType) }
                    else if (scaleName == "full") { EvalScale.full(s.vectorType) }
                    else { EvalScale.partial(s.vectorType) }
        println("\n" + "#" * 72)
        println(">>> 数据集: ${s.label}; scale=${scaleName}")
        println("#" * 72)
        runFullBenchmark(s.txtPath, s.label, scale)
    }
}

public func runParityPreset(selection: String, maxQueries: Int64, beta: Float64): Unit {
    let specOpt = findSpec(selection)
    if (specOpt.isNone()) { throw IllegalArgumentException("parity 仅支持单个数据集编号 1..6") }
    let spec = specOpt.getOrThrow()
    runParityTrace(spec.txtPath, selection, spec.vectorType, maxQueries, beta)
}
