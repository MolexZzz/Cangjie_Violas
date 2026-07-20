package semanticvg_cj.bench

import std.collection.*
import std.fs.*
import std.core.*
import std.convert.*

public struct FoldRep {
    public let trainRep: Array<Float64>
    public let testRep: Array<Float64>

    public init(trainRep: Array<Float64>, testRep: Array<Float64>) {
        this.trainRep = trainRep
        this.testRep = testRep
    }
}

public class FolderData {
    public let folderName: String
    public let key: String
    public let dataType: String
    public let vectors: ArrayList<Array<Float64>>
    public let descriptions: ArrayList<String>
    public let globalReps: HashMap<String, Array<Float64>>
    public let foldReps: HashMap<String, ArrayList<FoldRep>>

    public init(folderName: String, key: String, dataType: String) {
        this.folderName = folderName
        this.key = key
        this.dataType = dataType
        this.vectors = ArrayList<Array<Float64>>()
        this.descriptions = ArrayList<String>()
        this.globalReps = HashMap<String, Array<Float64>>()
        this.foldReps = HashMap<String, ArrayList<FoldRep>>()
    }
}

public class EvalScale {
    public let nFolds: Int64
    public let maxFolders: Int64
    public let maxQueries: Int64
    public let maxVectorsPerFolder: Int64
    public let topK: Int64
    public let vectorType: String
    public let repMethod: String

    public init(
        nFolds: Int64,
        maxFolders: Int64,
        maxQueries: Int64,
        maxVectorsPerFolder: Int64,
        topK: Int64,
        vectorType: String,
        repMethod: String
    ) {
        this.nFolds = nFolds
        this.maxFolders = maxFolders
        this.maxQueries = maxQueries
        this.maxVectorsPerFolder = maxVectorsPerFolder
        this.topK = topK
        this.vectorType = vectorType
        this.repMethod = repMethod
    }

    public static func partial(vectorType: String): EvalScale {
        // 每类最多加载 120 条向量，避免数百 MB TXT 一次性进内存
        EvalScale(5, 0, 200, 120, 3, vectorType, "simple")
    }

    public static func smoke(vectorType: String): EvalScale {
        // 小样本冒烟测试：配合预处理每类 20 条，检索查询也限制为 20 条
        EvalScale(5, 0, 20, 20, 3, vectorType, "simple")
    }

    public static func full(vectorType: String): EvalScale {
        EvalScale(5, 0, 0, 0, 3, vectorType, "simple")
    }
}

func parseLine(
    trimmed: String,
    dataset: HashMap<String, FolderData>,
    current: Option<FolderData>,
    maxVectorsPerFolder: Int64
): Option<FolderData> {
    if (trimmed.isEmpty()) {
        return current
    }
    let parts = trimmed.split("|")
    let cmd = parts[0]
    if (cmd == "FOLDER") {
        let typ = if (parts.size > 3) { parts[3] } else { "DATA" }
        let fd = FolderData(parts[1], parts[2], typ)
        dataset[parts[1]] = fd
        return Some(fd)
    }
    let fdOpt = current
    if (fdOpt.isNone()) {
        return current
    }
    let fd = fdOpt.getOrThrow()
    if (cmd == "VECTOR") {
        if (maxVectorsPerFolder > 0 && fd.vectors.size >= maxVectorsPerFolder) {
            return current
        }
        let vecStrs = parts[2].split(",")
        let vec = Array<Float64>(vecStrs.size, { i => Float64.parse(vecStrs[i]) })
        fd.descriptions.add(parts[1])
        fd.vectors.add(vec)
    } else if (cmd == "GLOBAL_REP") {
        if (parts.size > 2 && !parts[2].isEmpty()) {
            let repStrs = parts[2].split(",")
            let rep = Array<Float64>(repStrs.size, { i => Float64.parse(repStrs[i]) })
            fd.globalReps[parts[1]] = rep
        }
    } else if (cmd == "FOLD_REP") {
        let trRep = if (parts.size > 3 && !parts[3].isEmpty()) {
            let trStrs = parts[3].split(",")
            Array<Float64>(trStrs.size, { i => Float64.parse(trStrs[i]) })
        } else { Array<Float64>() }
        let teRep = if (parts.size > 4 && !parts[4].isEmpty()) {
            let teStrs = parts[4].split(",")
            Array<Float64>(teStrs.size, { i => Float64.parse(teStrs[i]) })
        } else { Array<Float64>() }
        if (!fd.foldReps.contains(parts[1])) {
            fd.foldReps[parts[1]] = ArrayList<FoldRep>()
        }
        fd.foldReps[parts[1]].add(FoldRep(trRep, teRep))
    }
    return current
}

func processCarryLine(
    line: String,
    dataset: HashMap<String, FolderData>,
    current: Option<FolderData>,
    maxVectorsPerFolder: Int64
): Option<FolderData> {
    parseLine(line.replace("\r", ""), dataset, current, maxVectorsPerFolder)
}

public func loadPrecomputedDataset(filePath: String, maxVectorsPerFolder: Int64): HashMap<String, FolderData> {
    println("正在加载(流式): ${filePath}")
    if (maxVectorsPerFolder > 0) {
        println("  每类最多加载 ${maxVectorsPerFolder} 条向量")
    }
    let dataset = HashMap<String, FolderData>()
    var current: Option<FolderData> = None
    let f = File(filePath, OpenMode.Read)
    let bufSize: Int64 = 65536
    let buf = Array<Byte>(bufSize, repeat: 0)
    var carry = ""

    while (true) {
        let n = f.read(buf)
        if (n <= 0) {
            break
        }
        let chunkBytes = Array<Byte>(n, { i => buf[i] })
        let chunk = String.fromUtf8(chunkBytes)
        carry = carry + chunk
        while (true) {
            match (carry.indexOf("\n")) {
                case Some(nl) =>
                    let line = carry[0..nl]
                    carry = carry[(nl + 1)..carry.size]
                    current = processCarryLine(line, dataset, current, maxVectorsPerFolder)
                case None => break
            }
        }
    }
    if (!carry.isEmpty()) {
        current = processCarryLine(carry, dataset, current, maxVectorsPerFolder)
    }
    println("加载完成，类别数: ${dataset.size}")
    return dataset
}

public func limitDataset(
    dataset: HashMap<String, FolderData>,
    maxFolders: Int64
): HashMap<String, FolderData> {
    if (maxFolders <= 0 || Int64(dataset.size) <= maxFolders) {
        return dataset
    }
    let limited = HashMap<String, FolderData>()
    var n: Int64 = 0
    for ((name, data) in dataset) {
        if (n >= maxFolders) { break }
        limited[name] = data
        n++
    }
    return limited
}
