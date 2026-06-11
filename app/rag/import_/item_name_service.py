from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import ITEM_NAME_CONTEXT_CHUNK_K, ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log, logger


@step_log("validate_chunks_and_title")
def validate_chunks_and_title(state) -> tuple[list[dict], str]:
    chunks = state.get("chunks")
    file_title = state.get("file_title")

    if not chunks:
        logger.error(f"chunks内容为空，无法继续业务！！")
        raise ValueError(f"chunks内容为空，无法继续业务！！")
    if not file_title:
        file_title = chunks[0]['file_title'] or "default_file_title"

    return chunks, file_title


@step_log("")
def build_document_context(chunks) -> str:
    # 上下文拼接

    # 1. 截取 topK chunk 内容
    top_chunk = chunks[:ITEM_NAME_CONTEXT_CHUNK_K]
    # 2. 拼接上下文 ->   切片: 1 标题: x  父标题: x  内容:  x \n
    context = ""
    for index, chunk in enumerate(top_chunk, start=1):
        context += f"切片:{index} 标题:{chunk['title']} 父标题: {chunks['parent_title']} 内容: {chunk['content']} \n"
    # 3. 最大的字符串长度限制
    final_context = context[: ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]
    return final_context


# 调用llm解析topk chunk_context获取主体名称item_name
@step_log("recognize_item_name")
def recognize_item_name(context: str, file_title: str) -> str:
    llm_client = llm_provider.chat()
    # 加载外部提示词
    system_prompt_str = load_prompt("product_recognition_system")
    human_prompt_str = load_prompt(
        "item_name_recognition",
        file_title = file_title,
        context = context
    )

    messages = [
        SystemMessage(content=system_prompt_str),
        HumanMessage(content=human_prompt_str)
    ]
    # 调用模型识别主体
    item_name = (llm_client | StrOutputParser()).invoke(messages)
    logger.info(f"调用模型进行item_name识别完毕！ item_name:{item_name}")
    # 非空判断和兜底 赋值
    if not item_name:
        item_name = file_title

    return item_name


@step_log("apply_item_name")
def apply_item_name(chunks: list[dict], item_name: str):
    # 给chunk赋值item_name
    for chunk in chunks:
        chunk['item_name'] = item_name
    logger.info(f"完成chunks的item_name数据补充！ {chunks[0]['item_name']}")


@step_log("embed_item_name")
def embed_item_name(item_name: str):
    # 根据item_name生成稠密和稀疏向量
    result = llm_provider.embed_documents([item_name])
    return result['dense'][0], result['sparse'][0]


@step_log("prepare_item_name_collection")
def prepare_item_name_collection():
    # 1. 获取客户端对象
    milvus_client = milvus_gateway.client
    # 2. 判断集合是否已存在
    if milvus_client.has_collection(collection_name=milvus_gateway.item_collection_name):
        logger.info(f"{milvus_gateway.item_collection_name}对应的集合存在，无需创建！")
        return

    # 3. 创建集合schema  添加field
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True
    )

    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length=512)
    schema.add_field(field_name="item_name",datatype=DataType.VARCHAR,max_length=512)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 4. 创建集合对应index（索引）
    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        # 给哪个字段创建索引 字段应该是经常查询的字段
        field_name="dense_vector",
        # 索引的类型 索引就是外部创建一种高效的数据类型  [目录]-> 查询 -> 内存地址 -> 链接到对应的实体数据
        # 推荐: AUTOINDEX -> 自动创建索引 自动选择类型 我有点不推荐!
        # 为了减少学习曲线，Milvus 提供了AUTOINDEX。通过AUTOINDEX，Milvus 可以在建立索引的同时分析 Collections
        # 中的数据分布，并根据分析结果设置最优化的索引参数，从而在搜索性能和正确性之间取得平衡。
        # HNSW : 分层图 -> 类似地图搜索过程  [精度最高 / 内存在有最大]
        # IVF_FLAT : 分桶 nlist = 64 找到对应桶 / 细化筛选 [比 FLAT快, 占有内存中等]
        # FLAT :  直接所有向量搜索和比较 [最慢]
        index_type="HNSW",
        # 相识度算法 L2 [0-2] COSINE  IP  [-1 1]
        metric_type="COSINE" ,
        params = {
            "M": 64,  # Maximum number of neighbors each node can connect to in the graph
            "efConstruction": 100  # Number of candidate neighbors considered for connection during index construction
        }  # I
    )

    index_params.add_index(
        field_name="sparse_vector",
        # 稀疏向量 2.6 只有倒排索引
        # 内容 -> 向量相似度
        # doc1 = {1:x 3:x}
        # doc2 = {1:x,4:x}
        # 1位置 = doc1 , doc2
        # 3位置 = doc1
        # 4位置 = doc2
        # 搜索的稀疏向量 {1:k} -> doc1 doc2
        index_type="SPARSE_INVERTED_INDEX",
        # IP (内积）：使用点积衡量相似性。
        metric_type="IP",
        # 算法识别 影响小的值跳过,提高相似度比较的效率
        params={"inverted_index_algo": "DAAT_MAXSCORE"}
    )

    # 5. 创建集合 （集合的名字 schema indexs)
    milvus_client.create_collection(
        collection_name=milvus_gateway.item_collection_name,
        schema=schema,
        index_params=index_params
    )
    logger.info(f"{milvus_gateway.item_collection_name}第一次完成初始化！！")


@step_log("upsert_item_name")
def upsert_item_name(item_name: str, file_title: str, dense_vector:list[float], sparse_vector: list[float]):
    # 先删除 / 再插入 幂等性
    milvus_client = milvus_gateway.client
    # 1. 先根据file_title 删除
    milvus_client.delete(
        collection_name=milvus_gateway.item_collection_name,
        filter=f"file_title == '{file_title}'"
    )
    # 2. 插入新的数据即可
    result = milvus_client.insert(
        collection_name=milvus_gateway.item_collection_name,
        data=[{
            "item_name": item_name,
            "file_title": file_title,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector
        }]
    )

    logger.info(f"{item_name}对应的数据已经插入到{milvus_gateway.item_collection_name}对应的集合中! 返回结果:{result}")


# 主业务入口
@step_log("recognize_and_index_item_name")
def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 item_name
    3. 将 item_name 回填到 state 和 chunks
    4. 同步向milvus中指定collection写入主体名称索引
    """



@step_log("recognize_and_index_item_name")
def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 item_name
    3. 将 item_name 回填到 state 和 chunks
    4. milvus同步写入主体名称索引
    """
    # 1. 参数校验
    chunks, file_title = validate_chunks_and_title(state)
    # 2. 进行上下文的拼接 chunks
    context = build_document_context(chunks)
    # 3. 进行item_name的识别
    item_name = recognize_item_name(context)
    # 4. 给chunks中chunk赋值item_name属性
    apply_item_name(item_name)
    # 5. 对item_name进行向量化，生成稠密和稀疏向量
    dense_vector, sparse_vector = embed_item_name(item_name)
    # 6. 准备item_name对应的集合信息
    prepare_item_name_collection()
    # 7. 更新或者存储item_name到对应集合
    upsert_item_name(item_name, file_title, dense_vector, sparse_vector)
    # 8. 更新state数据
    # state['chunks'] = chunks
    state['item_name'] = item_name
    return state