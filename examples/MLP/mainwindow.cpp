#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QVBoxLayout>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow),
      draw_widget(new DrawWidget(parent)) {
  ui->setupUi(this);
  ui->centralWidget->setLayout(new QVBoxLayout);
  draw_widget->setMaximumSize(280, 280);
  ui->centralWidget->layout()->addWidget(draw_widget);
  auto font = ui->plainTextEdit->font();
  font.setPointSize(30);
  ui->plainTextEdit->setFont(font);
  QObject::connect(ui->pushButton, &QPushButton::clicked, draw_widget,
                   &DrawWidget::clear);
  QObject::connect(draw_widget, &DrawWidget::predictNumber, [=](int num) {
    ui->plainTextEdit->document()->setPlainText(QString("%1").arg(num));
  });
}

MainWindow::~MainWindow() {
  delete ui;
  delete draw_widget;
}
