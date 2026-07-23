package semanticvg_cj.storage

import std.collection.*

/**
 * 根据输入字符串生成一个 64 位哈希值的十六进制字符串表示。
 * * @param input 用于生成 ID 的原始字符串
 * @return 长度为 16 的十六进制字符串
 */
private func generateGroupId(input: String): String {
    // 1. 获取字符串内置的 64 位整数哈希值
    var temp: Int64 = input.hashCode()
    
    // 2. 原生位运算：将 Int64 的每 4 位映射为一个十六进制字符
    var result = ""
    let hexChars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]
    
    var i = 0
    while (i < 16) {
        // 取最后 4 位作为索引 (0-15)
        let index = temp & 0x0F
        // 拼接字符串，注意这里是从后往前取，拼接在 result 前面
        result = hexChars[index] + result
        // 无符号右移 4 位
        temp = temp >> 4
        i++
    }
    return result
}

/**
 * VectorGroup 类，用于组织和管理一组相关的向量及其对应的元数据描述。
 * * 该类保证了组内所有向量的一致性（如维度相同），并自动生成唯一的组 ID。
 */
public class VectorGroup {
    // 基础属性
    public let groupName: String         // 向量组名称
    public let representative: Array<Float64> // 代表向量
    public let repDescription: String    // 代表向量的描述
    public let vectorType: String        // 向量类型
    public let groupType: String         // 分组类型
    public let groupId: String           // 基于组属性生成的唯一标识符
    
    // 数据存储
    public var vectors: ArrayList<Array<Float64>>     // 向量列表
    public var descriptions: ArrayList<HashMap<String, Any>> // 与向量一一对应的描述信息列表
    
    /**
     * 构造函数：初始化向量组并进行基础校验。
     * * @param groupName 组名
     * @param representative 代表向量
     * @param repDescription 代表向量描述
     * @param vectors 初始向量数组
     * @param descriptions 初始描述数组（需与 vectors 长度一致）
     * @param vectorType 向量分类标识，默认为 "general"
     * @param groupType 组分类标识，默认为 "default"
     */
    public init(
        groupName: String,
        representative: Array<Float64>,
        repDescription: String,
        vectors: Array<Array<Float64>>,
        descriptions: Array<Any>,
        vectorType!: String = "general",
        groupType!: String = "default"
    ) {
        // 1. 参数合法性校验
        if (vectors.size != descriptions.size) { 
            throw Exception("初始化失败：向量数量与描述数量不匹配。") 
        }
        if (vectors.isEmpty()) { 
            throw Exception("初始化失败：向量组不能为空。") 
        }
        if (representative.size != vectors[0].size) { 
            throw Exception("初始化失败：代表向量的维度 (${representative.size}) 与组内向量的维度 (${vectors[0].size}) 不匹配。") 
        }
        for (i in 0..vectors.size) {
            if (vectors[i].size != representative.size) {
                throw Exception(
                    "初始化失败：第 ${i} 个向量的维度 (${vectors[i].size}) 与代表向量维度 (${representative.size}) 不匹配。"
                )
            }
        }

        // 2. 赋值基础字段
        this.groupName = groupName
        this.representative = representative
        this.repDescription = repDescription
        this.vectorType = vectorType
        this.groupType = groupType
        
        // 3. 转换并存储向量数据
        this.vectors = ArrayList<Array<Float64>>(vectors)
        
        // 4. 标准化描述信息（假定存在外部定义的 normalizeDescriptions 函数）
        let normDescs = normalizeDescriptions(descriptions)
        this.descriptions = ArrayList<HashMap<String, Any>>(normDescs)
        
        // 5. 生成唯一的组 ID (基于名称、向量类型和组类型)
        let content = "${groupName}_${vectorType}_${groupType}"
        this.groupId = generateGroupId(content)
    }

    /**
     * 向组内添加一个新的向量及其描述。
     * * @param vector 待添加的 Float64 数组
     * @param description 对应的元数据
     * @return 返回添加后的索引位置，若维度不匹配则返回 -1
     */
    public func append(vector: Array<Float64>, description: Any): Int64 {
        // 检查维度是否与现有向量一致
        if (!this.vectors.isEmpty() && vector.size != this.vectors[0].size) {
            println("警告: 添加失败，向量维度不匹配: ${vector.size} != ${this.vectors[0].size}")
            return -1
        }
        
        this.vectors.add(vector)
        // 转换描述信息为标准 HashMap 格式（假定存在外部定义的 normalizeDescription 函数）
        this.descriptions.add(normalizeDescription(description))
        
        return this.vectors.size - 1
    }

    /** 原位更新一个对象；成功时向量和描述仍保持一一对应。 */
    public func updateAt(index: Int64, vector: Array<Float64>): Bool {
        if (index < 0 || index >= this.vectors.size) { return false }
        if (!this.vectors.isEmpty() && vector.size != this.vectors[0].size) { return false }
        this.vectors[index] = vector
        return true
    }

    /** 删除指定对象；调用方负责处理索引失效。 */
    public func removeAt(index: Int64): Bool {
        if (index < 0 || index >= this.vectors.size) { return false }
        this.vectors.remove(at: index)
        this.descriptions.remove(at: index)
        return true
    }

    /** 获取当前组内的向量总数 */
    public func size(): Int64 { return this.vectors.size }
    
    /** 判断向量组是否为空 */
    public func isEmpty(): Bool { return this.vectors.isEmpty() }

    /**
     * 获取向量组的统计摘要信息。
     * * @return 包含组名、ID、数量、维度等信息的 HashMap
     */
    public func getStatistics(): HashMap<String, Any> {
        let stats = HashMap<String, Any>()
        stats["group_name"] = this.groupName
        stats["group_id"] = this.groupId
        stats["vector_type"] = this.vectorType
        stats["group_type"] = this.groupType
        stats["vector_count"] = this.vectors.size
        // 如果组不为空，记录向量维度
        stats["vector_dimension"] = if (!this.isEmpty()) { this.vectors[0].size } else { 0 }
        stats["rep_description"] = this.repDescription
        return stats
    }
}
