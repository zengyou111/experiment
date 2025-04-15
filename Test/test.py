import ast
import javalang
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import networkx as nx
from collections import defaultdict
import numpy as np


@dataclass
class CodeSegment:
    """代码段类，表示一个具有完整语义的代码片段"""
    code: str
    start_line: int
    end_line: int
    segment_type: str  # 'method', 'class', 'function', 'block'
    parent_type: Optional[str] = None
    complexity: int = 0
    features: Dict = None


@dataclass
class CloneResult:
    """克隆检测结果类"""
    java_segment: CodeSegment
    python_segment: CodeSegment
    similarity: float
    clone_type: str  # Type-1, Type-2, Type-3
    matching_features: Dict


class SemanticCodeSplitter:
    """基于语义的代码分割器"""

    def __init__(self):
        self.min_segment_lines = 5
        self.max_segment_lines = 50

    def _get_method_lines(self, code: str, method) -> str:
        """获取方法的代码行"""
        lines = code.split('\n')
        method_start = method.position.line - 1  # Convert to 0-based index

        # Find the method end by matching brackets
        bracket_count = 0
        method_end = method_start

        for i in range(method_start, len(lines)):
            line = lines[i]
            bracket_count += line.count('{') - line.count('}')
            if bracket_count == 0:
                method_end = i
                break

        return '\n'.join(lines[method_start:method_end + 1])

    def _find_end_line(self, code: str, start_line: int) -> int:
        """计算代码段的结束行号"""
        return start_line + len(code.split('\n')) - 1

    def _get_python_function_code(self, code: str, node: ast.FunctionDef) -> str:
        """获取Python函数的代码"""
        lines = code.split('\n')
        return '\n'.join(lines[node.lineno - 1:node.end_lineno])

    def _get_python_class_code(self, code: str, node: ast.ClassDef) -> str:
        """获取Python类的代码"""
        lines = code.split('\n')
        return '\n'.join(lines[node.lineno - 1:node.end_lineno])

    def _fallback_split_java(self, code: str) -> List[CodeSegment]:
        """当Java解析失败时的备用分割方法"""
        segments = []
        lines = code.split('\n')
        current_segment = []
        current_indent = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # Check for method or class declarations
            if re.match(r'^(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(', stripped_line):
                if current_segment and len(current_segment) >= self.min_segment_lines:
                    segments.append(CodeSegment(
                        code='\n'.join(current_segment),
                        start_line=start_line,
                        end_line=i - 1,
                        segment_type='method',
                        complexity=self._calculate_complexity('\n'.join(current_segment))
                    ))
                current_segment = [line]
                start_line = i
                current_indent = len(line) - len(stripped_line)
            elif len(line) - len(stripped_line) > current_indent:
                current_segment.append(line)
            elif len(line) - len(stripped_line) <= current_indent and current_segment:
                if len(current_segment) >= self.min_segment_lines:
                    segments.append(CodeSegment(
                        code='\n'.join(current_segment),
                        start_line=start_line,
                        end_line=i - 1,
                        segment_type='block',
                        complexity=self._calculate_complexity('\n'.join(current_segment))
                    ))
                current_segment = [line]
                start_line = i
                current_indent = len(line) - len(stripped_line)
            else:
                current_segment.append(line)

        # Add the last segment if it exists
        if current_segment and len(current_segment) >= self.min_segment_lines:
            segments.append(CodeSegment(
                code='\n'.join(current_segment),
                start_line=start_line,
                end_line=len(lines),
                segment_type='block',
                complexity=self._calculate_complexity('\n'.join(current_segment))
            ))

        return segments

    def _fallback_split_python(self, code: str) -> List[CodeSegment]:
        """当Python解析失败时的备用分割方法"""
        segments = []
        lines = code.split('\n')
        current_segment = []
        current_indent = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # Check for function or class declarations
            if re.match(r'^(def|class)\s+\w+', stripped_line):
                if current_segment and len(current_segment) >= self.min_segment_lines:
                    segments.append(CodeSegment(
                        code='\n'.join(current_segment),
                        start_line=start_line,
                        end_line=i - 1,
                        segment_type='function' if 'def' in current_segment[0] else 'class',
                        complexity=self._calculate_complexity('\n'.join(current_segment))
                    ))
                current_segment = [line]
                start_line = i
                current_indent = len(line) - len(stripped_line)
            elif line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > current_indent:
                    current_segment.append(line)
                elif indent <= current_indent and current_segment:
                    if len(current_segment) >= self.min_segment_lines:
                        segments.append(CodeSegment(
                            code='\n'.join(current_segment),
                            start_line=start_line,
                            end_line=i - 1,
                            segment_type='block',
                            complexity=self._calculate_complexity('\n'.join(current_segment))
                        ))
                    current_segment = [line]
                    start_line = i
                    current_indent = indent
                else:
                    current_segment.append(line)

        # Add the last segment if it exists
        if current_segment and len(current_segment) >= self.min_segment_lines:
            segments.append(CodeSegment(
                code='\n'.join(current_segment),
                start_line=start_line,
                end_line=len(lines),
                segment_type='block',
                complexity=self._calculate_complexity('\n'.join(current_segment))
            ))

        return segments

    def split_java_code(self, code: str) -> List[CodeSegment]:
        """分割Java代码为语义片段"""
        segments = []
        try:
            tree = javalang.parse.parse(code)
            # 处理类级别
            for cls in tree.types:
                if isinstance(cls, javalang.tree.ClassDeclaration):
                    # 处理方法级别
                    for method in cls.methods:
                        if method.body:
                            # 获取方法的代码
                            method_lines = self._get_method_lines(code, method)
                            if len(method_lines.split('\n')) >= self.min_segment_lines:
                                segments.append(CodeSegment(
                                    code=method_lines,
                                    start_line=method.position.line,
                                    end_line=self._find_end_line(method_lines, method.position.line),
                                    segment_type='method',
                                    parent_type='class',
                                    complexity=self._calculate_complexity(method_lines)
                                ))

                            # 处理方法内的代码块
                            blocks = self._extract_java_blocks(method_lines)
                            for block in blocks:
                                if len(block.split('\n')) >= self.min_segment_lines:
                                    segments.append(CodeSegment(
                                        code=block,
                                        start_line=method.position.line,
                                        end_line=self._find_end_line(block, method.position.line),
                                        segment_type='block',
                                        parent_type='method',
                                        complexity=self._calculate_complexity(block)
                                    ))
        except:
            # 如果解析失败，尝试基于语法特征分割
            segments.extend(self._fallback_split_java(code))

        return segments

    def split_python_code(self, code: str) -> List[CodeSegment]:
        """分割Python代码为语义片段"""
        segments = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # 处理函数定义
                if isinstance(node, ast.FunctionDef):
                    func_code = self._get_python_function_code(code, node)
                    if len(func_code.split('\n')) >= self.min_segment_lines:
                        segments.append(CodeSegment(
                            code=func_code,
                            start_line=node.lineno,
                            end_line=node.end_lineno,
                            segment_type='function',
                            complexity=self._calculate_complexity(func_code)
                        ))

                    # 处理函数内的代码块
                    blocks = self._extract_python_blocks(func_code)
                    for block in blocks:
                        if len(block.split('\n')) >= self.min_segment_lines:
                            segments.append(CodeSegment(
                                code=block,
                                start_line=node.lineno,
                                end_line=self._find_end_line(block, node.lineno),
                                segment_type='block',
                                parent_type='function',
                                complexity=self._calculate_complexity(block)
                            ))

                # 处理类定义
                elif isinstance(node, ast.ClassDef):
                    class_code = self._get_python_class_code(code, node)
                    if len(class_code.split('\n')) >= self.min_segment_lines:
                        segments.append(CodeSegment(
                            code=class_code,
                            start_line=node.lineno,
                            end_line=node.end_lineno,
                            segment_type='class',
                            complexity=self._calculate_complexity(class_code)
                        ))

        except:
            # 如果解析失败，尝试基于缩进分割
            segments.extend(self._fallback_split_python(code))

        return segments

    def _calculate_complexity(self, code: str) -> int:
        """计算代码段的复杂度"""
        # 基于控制流程计算圈复杂度
        complexity = 1
        control_patterns = [
            r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b',
            r'\bcase\b', r'\bcatch\b', r'\b\|\|\b', r'\b&&\b'
        ]
        for pattern in control_patterns:
            complexity += len(re.findall(pattern, code))
        return complexity

    def _extract_java_blocks(self, code: str) -> List[str]:
        """提取Java代码中的代码块"""
        blocks = []
        # 匹配for, while, if等语句块
        block_pattern = r'((?:if|for|while|try)\s*\([^)]*\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        for match in re.finditer(block_pattern, code):
            blocks.append(match.group())
        return blocks

    def _extract_python_blocks(self, code: str) -> List[str]:
        """提取Python代码中的代码块"""
        blocks = []
        lines = code.split('\n')
        current_block = []
        current_indent = 0

        for line in lines:
            indent = len(line) - len(line.lstrip())
            if not current_block:
                if any(keyword in line for keyword in ['if', 'for', 'while', 'try']):
                    current_block.append(line)
                    current_indent = indent
            else:
                if indent > current_indent:
                    current_block.append(line)
                else:
                    if len(current_block) >= self.min_segment_lines:
                        blocks.append('\n'.join(current_block))
                    current_block = []

        if current_block and len(current_block) >= self.min_segment_lines:
            blocks.append('\n'.join(current_block))

        return blocks


class FeatureExtractor:
    """特征提取器"""

    def __init__(self):
        self.feature_extractors = [
            self._extract_structural_features,
            self._extract_lexical_features,
            self._extract_syntactic_features
        ]

    def extract_features(self, segment: CodeSegment) -> Dict:
        """提取代码段的特征"""
        features = {}
        for extractor in self.feature_extractors:
            features.update(extractor(segment))
        return features

    def _extract_structural_features(self, segment: CodeSegment) -> Dict:
        """提取结构特征"""
        features = {
            'length': len(segment.code.split('\n')),
            'complexity': segment.complexity,
            'nesting_depth': self._calculate_nesting_depth(segment.code),
            'control_structures': self._count_control_structures(segment.code),
        }
        return {'structural': features}

    def _extract_lexical_features(self, segment: CodeSegment) -> Dict:
        """提取词法特征"""
        code = segment.code.lower()
        features = {
            'keywords': self._count_keywords(code),
            'operators': self._count_operators(code),
            'literals': self._count_literals(code),
            'identifiers': len(set(re.findall(r'\b[a-zA-Z_]\w*\b', code)))
        }
        return {'lexical': features}

    def _extract_syntactic_features(self, segment: CodeSegment) -> Dict:
        """提取语法特征"""
        features = {
            'ast_depth': self._calculate_ast_depth(segment.code),
            'ast_size': self._calculate_ast_size(segment.code),
            'statement_types': self._count_statement_types(segment.code)
        }
        return {'syntactic': features}


class CloneDetector:
    """克隆检测器"""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.similarity_threshold = 0.8

    def detect_clones(self, java_segments: List[CodeSegment],
                      python_segments: List[CodeSegment]) -> List[CloneResult]:
        """检测代码克隆"""
        results = []

        # 提取特征
        for segment in java_segments + python_segments:
            segment.features = self.feature_extractor.extract_features(segment)

        # 构建相似度图
        graph = self._build_similarity_graph(java_segments, python_segments)

        # 使用社区检测算法找到克隆组
        communities = self._detect_communities(graph)

        # 处理每个社区
        for community in communities:
            java_members = [node for node in community if isinstance(node, tuple) and node[0] == 'java']
            python_members = [node for node in community if isinstance(node, tuple) and node[0] == 'python']

            # 对社区内的代码段进行两两比较
            for java_idx in java_members:
                for python_idx in python_members:
                    java_segment = java_segments[java_idx[1]]
                    python_segment = python_segments[python_idx[1]]

                    similarity = self._calculate_similarity(java_segment, python_segment)
                    if similarity >= self.similarity_threshold:
                        results.append(CloneResult(
                            java_segment=java_segment,
                            python_segment=python_segment,
                            similarity=similarity,
                            clone_type=self._determine_clone_type(java_segment, python_segment),
                            matching_features=self._get_matching_features(java_segment, python_segment)
                        ))

        return results

    def _build_similarity_graph(self, java_segments: List[CodeSegment],
                                python_segments: List[CodeSegment]) -> nx.Graph:
        """构建相似度图"""
        graph = nx.Graph()

        # 添加节点
        for i, segment in enumerate(java_segments):
            graph.add_node(('java', i))
        for i, segment in enumerate(python_segments):
            graph.add_node(('python', i))

        # 添加边
        for i, java_segment in enumerate(java_segments):
            for j, python_segment in enumerate(python_segments):
                similarity = self._calculate_similarity(java_segment, python_segment)
                if similarity >= self.similarity_threshold:
                    graph.add_edge(('java', i), ('python', j), weight=similarity)

        return graph

    def _detect_communities(self, graph: nx.Graph) -> List[Set]:
        """使用社区检测算法"""
        # 使用Louvain算法进行社区检测
        communities = nx.community.louvain_communities(graph, weight='weight')
        return communities

    def _calculate_similarity(self, segment1: CodeSegment, segment2: CodeSegment) -> float:
        """计算两个代码段的相似度"""
        weights = {
            'structural': 0.4,
            'lexical': 0.3,
            'syntactic': 0.3
        }

        total_similarity = 0
        for feature_type, weight in weights.items():
            features1 = segment1.features[feature_type]
            features2 = segment2.features[feature_type]
            similarity = self._calculate_feature_similarity(features1, features2)
            total_similarity += weight * similarity

        return total_similarity

    def _determine_clone_type(self, segment1: CodeSegment, segment2: CodeSegment) -> str:
        """确定克隆类型"""
        normalized1 = self._normalize_code(segment1.code)
        normalized2 = self._normalize_code(segment2.code)

        if normalized1 == normalized2:
            return 'Type-1'  # 完全相同或仅有空白字符、注释差异

        tokenized1 = self._tokenize_code(normalized1)
        tokenized2 = self._tokenize_code(normalized2)

        if tokenized1 == tokenized2:
            return 'Type-2'  # 结构相同但变量名等可能不同

        return 'Type-3'  # 存在语句级别的修改


def main():
    """主函数示例"""
    # 示例代码
    java_code = """
    public class Calculator {
        public int factorial(int n) {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }

        public int fibonacci(int n) {
            if (n <= 1) return n;
            int a = 0, b = 1;
            for (int i = 2; i <= n; i++) {
                int temp = b;
                b = a + b;
                a = temp;
            }
            return b;
        }
    }
    """

    python_code = """
    class Calculator:
        def factorial(self, n):
            if n <= 1:
                return 1
            return n * self.factorial(n - 1)

        def fibonacci(self, n):
            if n <= 1:
                return n
            a, b = 0, 1
            for i in range(2, n + 1):
                a, b = b, a + b
            return b
    """

    # 创建分析器
    splitter = SemanticCodeSplitter()
    detector = CloneDetector()



    # 分割代码
    java_segments = splitter.split_java_code(java_code)
    python_segments = splitter.split_python_code(python_code)

    # 检测克隆
    results = detector.detect_clones(java_segments, python_segments)
if __name__ == '__main__':
    main()