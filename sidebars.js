// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  studySidebar: [
    {
      type: 'doc',
      id: 'TCX3212-PA/index',
      label: 'TCX3212 Predictive Analytics',
    },
    {
      type: 'category',
      label: '📚 Lectures',
      collapsible: true,
      collapsed: false,
      items: [
        'TCX3212-PA/lectures/l0-overview',
        'TCX3212-PA/lectures/l1-introduction',
        'TCX3212-PA/lectures/l2-python-basics-1',
        'TCX3212-PA/lectures/l3-python-basics-2',
        'TCX3212-PA/lectures/l4-data-preparation',
        'TCX3212-PA/lectures/l5-regression-1',
        'TCX3212-PA/lectures/l6-regression-2',
        'TCX3212-PA/lectures/l7-logistic-regression',
        'TCX3212-PA/lectures/l8-time-series-1',
        'TCX3212-PA/lectures/l9-time-series-2',
        'TCX3212-PA/lectures/l10-time-series-3',
      ],
    },
    {
      type: 'category',
      label: '🧪 Tutorials',
      collapsible: true,
      collapsed: false,
      items: [
        'TCX3212-PA/tutorials/t1',
        'TCX3212-PA/tutorials/t1-solution',
        'TCX3212-PA/tutorials/t2',
        'TCX3212-PA/tutorials/t2-solution',
        'TCX3212-PA/tutorials/t3',
        'TCX3212-PA/tutorials/t3-solution',
        'TCX3212-PA/tutorials/t4',
        'TCX3212-PA/tutorials/t4-solution',
        'TCX3212-PA/tutorials/t5',
        'TCX3212-PA/tutorials/t5-solution',
        'TCX3212-PA/tutorials/t6',
        'TCX3212-PA/tutorials/t6-solution',
        'TCX3212-PA/tutorials/t7',
        'TCX3212-PA/tutorials/t7-solution',
        'TCX3212-PA/tutorials/t8',
        'TCX3212-PA/tutorials/t8-solution',
        'TCX3212-PA/tutorials/t9',
        'TCX3212-PA/tutorials/t9-solution',
      ],
    },
    {
      type: 'category',
      label: '📝 Assessments',
      collapsible: true,
      collapsed: false,
      items: [
        'TCX3212-PA/assessments/assignment-1',
        'TCX3212-PA/assessments/assignment-2',
        'TCX3212-PA/assessments/group-project',
      ],
    },
  ],
};

export default sidebars;
