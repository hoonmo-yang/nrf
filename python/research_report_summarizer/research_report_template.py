from __future__ import annotations
from pydantic import BaseModel, Field


class ResearchReportSummaryTemplate(BaseModel):
    class Config:
        title = "연구개발 과제 결과 요약문"
        description = "연구개발 과제 결과 요약문입니다"

    abstract: str = Field(
        ...,
        description="연구개요",
    )

    outcome: str = Field(
        ...,
        description="연구 목표대비 연구결과",
    )

    expectation: str = Field(
        ...,
        description="연구개발성과의 활용 계획 및 기대효과(연구개발결과의 중요성)",
    )

    keyword_kr: list[str] = Field(
        ...,
        description="총 5개의 중심어 (국문)",
    )

    keyword_en: list[str] = Field(
        ...,
        description="총 5개의 중심어 (영문)",
    )


class ResearchReportItem(BaseModel):
    class Config:
        title = "연구개발 과제 항목 기본 클래스"
        description = "연구개발 과제 항목 기본 클래스"

    error: bool = False


class ResearchReportItemSummary(ResearchReportItem):
    class Config:
        title = "내용 요약 결과"
        description = "내용 요약 결과 (abstract, outcome, expectation)"

    summary: str = Field(
        ...,
        description="내용 요약 결과",
    )


class ResearchReportItemKeyword(ResearchReportItem):
    class Config:
        title = "중심어 추출 결과"
        description = "중심어 추출 결과 (keyword)"

    keyword_kr: list[str] = Field(
        ...,
        description="국문 중심어 추출 결과",
    )

    keyword_en: list[str] = Field(
        ...,
        description="영문 중심어 추출 결과",
    )


class ResearchReportItemFail(ResearchReportItem):
    class Config:
        title = "결과 추출 실패"
        description = "결과 추출 실패"

    message: str = Field(
        ...,
        description="결과 추출 실패 사유"
    )

    error: bool = True


class ResearchReportIdentity(BaseModel):
    class Config:
        title = "연구개발과제 정보"
        description = "연구개발과제 정보"

    title_kr: str = Field(
        ...,
        description="연구개발과제명 (국문)",
    )

    title_en: str = Field(
        ...,
        description="연구개발과제명 (영문)",
    )

    institution: str = Field(
        ...,
        description="주관연구개발기관"
    )

    name: str = Field(
        ...,
        description="저자 성명"
    )

    position: str = Field(
        ...,
        description="저자 직위(직급)"
    )

    department: str = Field(
        ...,
        description="저자 소속"
    )

    major: str = Field(
        ...,
        description="저자 전공 분야"
    )


class ResearchReportSimilarity(BaseModel):
    class Config:
        title = "개발연구과제 보고서 요약 항목별 유사도 점수"
        description = "개발연구과제 보고서 요약 항목별 유사도 점수"

    abstract_score: float = Field(
        ...,
        description="abstract 항목 유사도 점수 (10점 만점)",
    )

    abstract_difference: str = Field(
        ...,
        description="abstract 항목의 내용이 다른 부분에 대한 설명",
    )

    outcome_score: float = Field(
        ...,
        description="outcome 항목 유사도 점수 (10점 만점)",
    )

    outcome_difference: str = Field(
        ...,
        description="outcome 항목의 내용이 다른 부분에 대한 설명",
    )

    expectation_score: float = Field(
        ...,
        description="expectation 항목 유사도 점수 (10점 만점)",
    )

    expectation_difference: str = Field(
        ...,
        description="expectation 항목의 내용이 다른 부분에 대한 설명",
    )

    keyword_score: float = Field(
        ...,
        description="keyword 항목 유사도 점수 (10점 만점)",
    )

    keyword_difference: str = Field(
        ...,
        description="keyword 항목의 내용이 다른 부분에 대한 설명",
    )
