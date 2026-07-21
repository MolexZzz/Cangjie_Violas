package semanticvg_cj.storage

import std.convert.*

public func clampUnit(value: Float64): Float64 {
    if (value < 0.0) { return 0.0 }
    if (value > 1.0) { return 1.0 }
    return value
}

public func mixedDistance(embeddingDistance: Float64, semanticDistance: Option<Float64>, beta: Float64): Float64 {
    let weight = clampUnit(beta)
    return match (semanticDistance) {
        case Some(semantic) => weight * semantic + (1.0 - weight) * embeddingDistance
        case None => embeddingDistance
    }
}

public func storageBaseKey(key: String): String {
    let idxOpt = key.lastIndexOf("-")
    if (idxOpt.isNone()) { return key }
    let idx = idxOpt.getOrThrow()
    if (idx <= 0 || idx >= key.size - 1) { return key }
    let suffix = key[(idx + 1)..key.size]
    try {
        let _ = Int64.parse(suffix)
        return key[0..idx]
    } catch (e: Exception) {
        return key
    }
}
