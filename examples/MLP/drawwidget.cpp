#include "drawwidget.h"
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QRgb>

#include "model.h"

#include <mgcpp/mgcpp.hpp>
#include <mgcpp/operations/trans.hpp>

template <size_t NIn,
          size_t NOut,
          float (&Weights)[NIn][NOut],
          float (&Bias)[NOut]>
struct Layer {
  using matrix = mgcpp::device_matrix<float>;
  using vector = mgcpp::device_vector<float>;
  matrix W;
  vector b;

  Layer() {
    W = mgcpp::strict::trans(matrix::from_c_array(Weights));
    b = vector::from_c_array(Bias);
  }

  template <typename T>
  auto operator()(T const& input) {
    return mgcpp::ref(W) * input + mgcpp::ref(b);
  }
};

DrawWidget::DrawWidget(QWidget* parent) : QWidget(parent) {}

DrawWidget::~DrawWidget() {}

void DrawWidget::drawPixel(QPoint pt) {
  if (0 <= pt.x() && pt.x() < width() && 0 <= pt.y() && pt.y() < height()) {
    QPainter p(&m_canvas);
    p.setRenderHints(QPainter::HighQualityAntialiasing);
    p.setPen(QPen(QColor(0, 0, 0, 200), 2.0));
    p.drawLine(last_point / 10, pt / 10);
    p.end();
  }
}

void DrawWidget::mousePressEvent(QMouseEvent* event) {
  if (event->buttons() & Qt::LeftButton) {
    last_point = event->pos();
    repaint();
  }
}

void DrawWidget::predict() {
  std::vector<float> data(28 * 28);
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      data[i * 28 + j] = 1.f - (m_canvas.pixel(j, i) & 0xff) / 256.f;
    }
  }

  mgcpp::device_vector<float> input(28 * 28, data.data());

  Layer<28 * 28, 200, w1, b1> l_input;
  auto y1 = mgcpp::relu(l_input(mgcpp::ref(input)));

  Layer<200, 100, w2, b2> l_hidden1;
  auto y2 = mgcpp::relu(l_hidden1(y1));

  Layer<100, 100, w3, b3> l_hidden2;
  auto y3 = mgcpp::relu(l_hidden2(y2));

  Layer<100, 10, w4, b4> l_output;
  auto result = l_output(y3);

  auto output = result.eval();

  std::vector<float> pr(10);
  for (int i = 0; i < 10; ++i)
    pr[i] = output.check_value(i);

  int answer = std::max_element(pr.begin(), pr.end()) - pr.begin();
  emit predictNumber(answer);
}

void DrawWidget::mouseMoveEvent(QMouseEvent* event) {
  if (event->buttons() & Qt::LeftButton) {
    drawPixel(event->pos());
    last_point = event->pos();
    predict();
    repaint();
  }
}

void DrawWidget::resizeEvent(QResizeEvent*) {
  m_canvas = QImage(28, 28, QImage::Format_RGBA8888);
  clear();
}

void DrawWidget::clear() {
  m_canvas.fill(0xFFFFFFFF);
  repaint();
}

void DrawWidget::paintEvent(QPaintEvent* event) {
  QWidget::paintEvent(event);
  QPainter painter(this);

  painter.drawPixmap(QRect(0, 0, 280, 280), QPixmap::fromImage(m_canvas),
                     QRect(0, 0, 28, 28));
}
