# Evaluation Datasets for Product Hunt RAG Analyzer

This directory contains evaluation datasets for validating the Product Hunt RAG Analyzer system.

## Overview

The evaluation framework consists of four components:

1. **Benchmark Queries** - Product ideas with known competitors for retrieval evaluation
2. **Labeled Reviews** - Sentiment-labeled reviews for sentiment analysis evaluation
3. **Feature Gaps** - Curated list of product features for feature extraction evaluation
4. **Annotation Guidelines** - Comprehensive guidelines for dataset creation and quality assurance

## Files

### 1. `benchmark_queries.jsonl`

**Purpose**: Evaluate RAG retrieval quality (competitor identification)

**Format**: JSONL (one JSON object per line)

**Structure**:
```json
{
  "product_idea": "A task management app with AI-powered prioritization...",
  "expected_competitors": ["Todoist", "TickTick", "Any.do", "Things 3", "Notion"],
  "category": "task_management"
}
```

**Statistics**:
- Total queries: 20
- Categories: 7 (task_management, note_taking, time_tracking, project_management, calendar, email, voice_dictation)
- Expected competitors per query: 3-5

**Evaluation Metrics**:
- Precision@5: Percentage of top-5 retrieved competitors that are relevant (target: ≥0.70)
- Recall@5: Percentage of expected competitors found in top-5 (target: ≥0.60)
- Mean Reciprocal Rank (MRR): Average reciprocal rank of first relevant result (target: ≥0.75)

### 2. `labeled_reviews.jsonl`

**Purpose**: Evaluate sentiment analysis accuracy

**Format**: JSONL (one JSON object per line)

**Structure**:
```json
{
  "review_id": "5012215",
  "body": "I've been an early user of Typeless, and it's hands down the best app...",
  "sentiment": "positive",
  "product_id": "1037362"
}
```

**Statistics**:
- Total reviews: 100
- Current distribution: 9% positive, 1% negative, 90% neutral
- Target distribution: 40% positive, 35% negative, 25% neutral

**Note**: The current distribution is heavily skewed toward neutral. You may want to manually re-label some reviews to achieve the target distribution for better evaluation coverage.

**Evaluation Metrics**:
- Classification Accuracy: Percentage of correctly classified reviews (target: ≥0.82)
- F1-Score: Macro-averaged F1 across sentiment classes (target: ≥0.80)

### 3. `feature_gaps.json`

**Purpose**: Evaluate feature extraction and gap analysis

**Format**: JSON (single object with array of features)

**Structure**:
```json
{
  "features": [
    {
      "feature": "Multi-language support",
      "category": "functionality",
      "priority": "high",
      "description": "Support for non-English languages (Spanish, French, German, etc.)"
    }
  ]
}
```

**Statistics**:
- Total features: 55
- Priority distribution: 44% high, 45% medium, 11% low
- Categories: 14 (functionality, ai_features, productivity, integrations, ui_ux, security, collaboration, analytics, organization, customization, support, deployment, platform, automation)

**Evaluation Metrics**:
- Extraction Recall: Percentage of known features extracted from reviews (target: ≥0.75)
- Categorization Accuracy: Percentage of correctly categorized features (target: ≥0.80)

### 4. `annotation_guidelines.md`

**Purpose**: Comprehensive guidelines for dataset annotation and quality assurance

**Contents**:
- Benchmark queries annotation process
- Sentiment analysis classification criteria
- Feature gaps identification and categorization
- Inter-rater agreement calculation (Cohen's Kappa)
- Quality assurance procedures
- Example annotations with reasoning

**Key Requirements**:
- Inter-rater agreement: Cohen's Kappa ≥ 0.80
- Annotation confidence: 80%+ high confidence
- Quality checks at pre, during, and post-annotation stages

## Usage

### Running Evaluations

```bash
# Run all evaluations
python -m src.evaluation.run_evaluation --full

# Run specific evaluation
python -m src.evaluation.validate_retrieval
python -m src.evaluation.validate_sentiment
python -m src.evaluation.validate_feature_gaps

# Generate evaluation report
python -m src.evaluation.generate_report --output reports/eval_report.html
```

### Evaluation Workflow

1. **Load Datasets**: Load evaluation datasets from this directory
2. **Run System**: Execute the RAG analyzer on benchmark queries
3. **Compare Results**: Compare system outputs with ground truth labels
4. **Calculate Metrics**: Compute evaluation metrics (Precision@5, Recall@5, MRR, Accuracy, F1)
5. **Generate Report**: Create comprehensive evaluation report with pass/fail status

### Success Criteria

The system is ready for use when:

1. **Retrieval Quality**:
   - Precision@5 ≥ 0.70
   - Recall@5 ≥ 0.60
   - MRR ≥ 0.75

2. **Sentiment Analysis**:
   - Accuracy ≥ 0.82
   - F1-Score ≥ 0.80

3. **Feature Gap Analysis**:
   - Extraction Recall ≥ 0.75
   - Categorization Accuracy ≥ 0.80

4. **System Performance**:
   - Average latency ≤ 30 seconds
   - Error rate ≤ 2%

5. **End-to-End Quality**:
   - Report completeness ≥ 0.95

## Data Quality Notes

### Current Issues

1. **Sentiment Distribution Imbalance**: 
   - Current: 9% positive, 1% negative, 90% neutral
   - Target: 40% positive, 35% negative, 25% neutral
   - **Action Required**: Manually re-label reviews to achieve target distribution

2. **Benchmark Query Competitors**:
   - Some expected competitors may not exist in your dataset
   - **Action Required**: Validate that all expected competitors are present in `dataset/products.jsonl`

### Recommendations

1. **Improve Sentiment Labels**:
   - Review the `labeled_reviews.jsonl` file
   - Re-classify reviews based on annotation guidelines
   - Aim for target distribution (40/35/25)

2. **Validate Benchmark Queries**:
   - Check that expected competitors exist in your product dataset
   - Update competitor lists if needed
   - Ensure queries are realistic and diverse

3. **Expand Feature Gaps**:
   - Add more features specific to your domain
   - Extract features from actual reviews in your dataset
   - Ensure features are extractable from text

4. **Calculate Inter-Rater Agreement**:
   - Have multiple annotators label a subset (20%)
   - Calculate Cohen's Kappa
   - Ensure κ ≥ 0.80 before proceeding

## Maintenance

### Updating Datasets

When updating evaluation datasets:

1. Follow annotation guidelines strictly
2. Maintain target distributions
3. Document changes in revision history
4. Re-run evaluations to ensure consistency
5. Update this README with new statistics

### Version History

| Version | Date | Changes | Annotator |
|---------|------|---------|-----------|
| 1.0 | 2024-12-10 | Initial datasets created | System |

## References

- **Annotation Guidelines**: See `annotation_guidelines.md` for detailed annotation procedures
- **Evaluation Framework**: See `.kiro/specs/product-hunt-rag-analyzer/design.md` for evaluation framework design
- **Requirements**: See `.kiro/specs/product-hunt-rag-analyzer/requirements.md` for evaluation requirements (Requirement 13)

## Contact

For questions about evaluation datasets:
- Review annotation guidelines
- Check evaluation framework documentation
- Consult with project maintainers
