from types import SimpleNamespace
import pytest

def ns(**kwargs):
    return SimpleNamespace(**kwargs)

@pytest.fixture
def simple_two_col_doc():
    def make_paragraph(start, end, x1, x2):
        return ns(
            layout=ns(
                text_anchor=ns(
                    text_segments=[ns(start_index=start, end_index=end)]
                ),
                bounding_poly=[
                    ns(x=x1, y=0.1),
                    ns(x=x2, y=0.1),
                    ns(x=x2, y=0.2),
                    ns(x=x1, y=0.2),
                ]
            ),
            type="paragraph",
            detected_languages=[]
        )

    return ns(
        text="LEFT1\nLEFT2\nRIGHT1\nRIGHT2\n",
        pages=[
            ns(
                paragraphs=[
                    make_paragraph(0, 5, 0.1, 0.3),   # LEFT1
                    make_paragraph(6, 11, 0.1, 0.3),  # LEFT2
                    make_paragraph(12, 18, 0.6, 0.9), # RIGHT1
                    make_paragraph(19, 25, 0.6, 0.9), # RIGHT2
                ],
                tables=[]
            )
        ]
    )


@pytest.fixture
def simple_table_doc():
    return ns(
        text="H1 H2 a1 a2 b1 b2 ",
        pages=[
            ns(
                blocks=[],
                tables=[
                    ns(
                        header_rows=[
                            ns(cells=[
                                ns(layout=ns(
                                    text_anchor=ns(text_segments=[
                                        ns(start_index=0, end_index=2)
                                    ]),
                                    bounding_poly=[
                                        ns(x=0.1, y=0.1),
                                        ns(x=0.3, y=0.1),
                                        ns(x=0.3, y=0.2),
                                        ns(x=0.1, y=0.2),
                                    ]
                                )),
                                ns(layout=ns(
                                    text_anchor=ns(text_segments=[
                                        ns(start_index=3, end_index=5)
                                    ]),
                                    bounding_poly=[
                                        ns(x=0.4, y=0.1),
                                        ns(x=0.6, y=0.1),
                                        ns(x=0.6, y=0.2),
                                        ns(x=0.4, y=0.2),
                                    ]
                                )),
                            ])
                        ],
                        body_rows=[
                            ns(cells=[
                                ns(layout=ns(
                                    text_anchor=ns(text_segments=[
                                        ns(start_index=6, end_index=8)
                                    ]),
                                    bounding_poly=[
                                        ns(x=0.1, y=0.3),
                                        ns(x=0.3, y=0.3),
                                        ns(x=0.3, y=0.4),
                                        ns(x=0.1, y=0.4),
                                    ]
                                )),
                                ns(layout=ns(
                                    text_anchor=ns(text_segments=[
                                        ns(start_index=9, end_index=11)
                                    ]),
                                    bounding_poly=[
                                        ns(x=0.4, y=0.3),
                                        ns(x=0.6, y=0.3),
                                        ns(x=0.6, y=0.4),
                                        ns(x=0.4, y=0.4),
                                    ]
                                )),
                            ])
                        ]
                    )
                ]
            )
        ]
    )
