package semanticvg_cj.storage

import std.math.*
import std.collection.*


// ============= 数学工具 =============

/* 计算向量的点积 */
func dotProduct(a: Array<Float64>, b: Array<Float64>): Float64 {
    if (a.size != b.size) { return 0.0 }
    var sum = 0.0
    for (i in 0..a.size) { sum += a[i] * b[i] }
    return sum
}

/* 计算向量的 L2 范数 */
func l2Norm(v: Array<Float64>): Float64 {
    return sqrt(dotProduct(v, v))
}

// ============= 距离和相似度计算函数 =============

public func cosineSimilarity(a: Array<Float64>, b: Array<Float64>): Float64 {
    let normA = l2Norm(a)
    let normB = l2Norm(b)
    if (abs(normA) < 1e-9 || abs(normB) < 1e-9) { return 0.0 }
    return dotProduct(a, b) / (normA * normB)
}

public func cosineDistance(a: Array<Float64>, b: Array<Float64>): Float64 {
    return 1.0 - cosineSimilarity(a, b)
}

public func euclideanDistance(a: Array<Float64>, b: Array<Float64>): Float64 {
    if (a.size != b.size) { return 0.0 }
    var sum = 0.0
    for (i in 0..a.size) {
        let diff = a[i] - b[i]
        sum += diff * diff
    }
    return sqrt(sum)
}

public func euclideanSimilarity(a: Array<Float64>, b: Array<Float64>): Float64 {
    return 1.0 / (1.0 + euclideanDistance(a, b))
}

public func manhattanDistance(a: Array<Float64>, b: Array<Float64>): Float64 {
    if (a.size != b.size) { return 0.0 }
    var sum = 0.0
    for (i in 0..a.size) { sum += abs(a[i] - b[i]) }
    return sum
}

public func manhattanSimilarity(a: Array<Float64>, b: Array<Float64>): Float64 {
    return 1.0 / (1.0 + manhattanDistance(a, b))
}

// ============= 向量列表计算函数 =============

public func hausdorffDistance(setA: Array<Array<Float64>>, setB: Array<Array<Float64>>): Float64 {
    if (setA.isEmpty() || setB.isEmpty()) { return Float64.Max }

    func directedDist(src: Array<Array<Float64>>, dst: Array<Array<Float64>>): Float64 {
        var maxMin = 0.0
        for (u in src) {
            var minD = Float64.Max
            for (v in dst) {
                minD = min(minD, euclideanDistance(u, v))
            }
            maxMin = max(maxMin, minD)
        }
        return maxMin
    }

    return max(directedDist(setA, setB), directedDist(setB, setA))
}

// hausdorff_similarity
public func hausdorffSimilarity(setA: Array<Array<Float64>>, setB: Array<Array<Float64>>): Float64 {
    let dist = hausdorffDistance(setA, setB)
    return 1.0 / (1.0 + dist)
}

public func maxSimilarity(
    setA: Array<Array<Float64>>, 
    setB: Array<Array<Float64>>, 
    simFunc: (Array<Float64>, Array<Float64>) -> Float64
): Float64 {
    if (setA.isEmpty() || setB.isEmpty()) { return 0.0 }
    var maxSim = 0.0
    for (a in setA) {
        for (b in setB) {
            maxSim = max(maxSim, simFunc(a, b))
        }
    }
    return maxSim
}

// ============= Description 规范化 =============

public func normalizeDescription(desc: Any): HashMap<String, Any> {
    match (desc) {
        // 1. 如果本来就是 HashMap，直接返回
        case d: HashMap<String, Any> => d
        
        // 2. 如果是字符串，塞进 "text" 键
        case s: String => 
            let map = HashMap<String, Any>()
            map["text"] = s  // ✅ 修复：使用下标语法代替 put
            map
            
        // 3. ✅ 修复：如果是 Option 类型，先判断，再展开内部的 None/Some
        case opt: Option<Any> => 
            match (opt) {
                case None => HashMap<String, Any>()
                case Some(val) => normalizeDescription(val) // 递归处理里面的真实值
            }
            
        // 4. 其他任意类型，塞进 "default" 键
        case _ => 
            let map = HashMap<String, Any>()
            map["default"] = desc // ✅ 修复：使用下标语法代替 put
            map
    }
}

// normalize_descriptions 批量转换
public func normalizeDescriptions(descs: Array<Any>): Array<HashMap<String, Any>> {
    let result = Array<HashMap<String, Any>>(descs.size, { _ => HashMap<String, Any>() })
    for (i in 0..descs.size) {
        result[i] = normalizeDescription(descs[i])
    }
    return result
}
