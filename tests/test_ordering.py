import pytest
from types import SimpleNamespace as NS
from src.ordering import reconstruct_boxes

@pytest.fixture
def simple_two_col_doc():
    return NS(
        text="LEFT1\nLEFT2\nRIGHT1\nRIGHT2\n",
        pages=[
            NS(
                paragraphs=[
                    NS(
                        layout=NS(
                            text_anchor=NS(text_segments=[
                                NS(start_index=0, end_index=5)
                            ]),
                            bounding_poly=NS(normalized_vertices=[
                                NS(x=0.1, y=0.1),
                                NS(x=0.3, y=0.1),
                                NS(x=0.3, y=0.2),
                                NS(x=0.1, y=0.2),
                            ])
                        ),
                        type="paragraph",
                        detected_languages=[]
                    ),
                    NS(
                        layout=NS(
                            text_anchor=NS(text_segments=[
                                NS(start_index=6, end_index=11)
                            ]),
                            bounding_poly=NS(normalized_vertices=[
                                NS(x=0.1, y=0.3),
                                NS(x=0.3, y=0.3),
                                NS(x=0.3, y=0.4),
                                NS(x=0.1, y=0.4),
                            ])
                        ),
                        type="paragraph",
                        detected_languages=[]
                    ),
                    NS(
                        layout=NS(
                            text_anchor=NS(text_segments=[
                                NS(start_index=12, end_index=18)
                            ]),
                            bounding_poly=NS(normalized_vertices=[
                                NS(x=0.6, y=0.1),
                                NS(x=0.9, y=0.1),
                                NS(x=0.9, y=0.2),
                                NS(x=0.6, y=0.2),
                            ])
                        ),
                        type="paragraph",
                        detected_languages=[]
                    ),
                    NS(
                        layout=NS(
                            text_anchor=NS(text_segments=[
                                NS(start_index=19, end_index=25)
                            ]),
                            bounding_poly=NS(normalized_vertices=[
                                NS(x=0.6, y=0.3),
                                NS(x=0.9, y=0.3),
                                NS(x=0.9, y=0.4),
                                NS(x=0.6, y=0.4),
                            ])
                        ),
                        type="paragraph",
                        detected_languages=[]
                    )
                ],
                tables=[]
            )
        ]
    )

def test_two_column_ordering(simple_two_col_doc):
    boxes = reconstruct_boxes(simple_two_col_doc)
    texts = [b.text for b in boxes]
    assert texts == ["LEFT1", "LEFT2", "RIGHT1", "RIGHT2"]

def test_handles_equal_y_uses_x_breaks(simple_two_col_doc):
    boxes = reconstruct_boxes(simple_two_col_doc)
    xs = [b.x0 for b in boxes]
    assert xs[0] < 0.5 and xs[1] < 0.5 and xs[2] > 0.5 and xs[3] > 0.5
