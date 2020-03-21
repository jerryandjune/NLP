//Ajax调用
function textchanged() {
  ProcessNews()
}

//加载随机数据
function LoadData() {
  $.ajax({
    url: "/LoadData",
    type: "Get",
    dataType: 'json',
    success: function (data) {
      console.log(data)
      $("#NewsTitle").val(data.result.NewsTitle)
      $("#NewsContent").val(data.result.NewsContent)
      $("#NewSummaryLength").val(data.result.NewSummaryLength)
      $.message({
        message: '加载成功！',
        type: 'success'
      });
    },
    error: function (e) {
      $.message({
        message: '处理失败！',
        type: 'error'
      });
    }
  })
}

//清除控件数据
function Clear() {
  $("#NewsTitle").val('')
  $("#NewsContent").val('')
  $("#NewSummary").val('')
  $("#NewSummaryLength").val('5')
  $.message({
    message: '清空成功！',
    type: 'success'
  });
}

//处理新闻
function ProcessNews() {
  error = false;
  content = $("#NewsContent").val()
  title = $("#NewsTitle").val()
  if (Number($("#NewSummaryLength").val()) <= 0) {
    $.message({
      message: '摘要长度必须大于0！',
      type: 'warning',
      duration: 0,
      center: true,
    });
    error = true;
  }

  if (content == '') {
    $.message({
      message: '请输入想要获取摘要的新闻！',
      type: 'warning',
      duration: 0,
      center: true,
    });
    error = true;
  }

  if (error) {
    return;
  }

  $.ajax({
    url: "/GetNewSummary",
    type: "Post",
    data:
    {
      NewsTitle: title,
      NewsContent: content,
      NewSummaryLength: $("#NewSummaryLength").val(),
    },
    dataType: 'json',
    success: function (data) {
      console.log(data)
      $("#NewSummary").val(data.result)
      $.message({
        message: '处理成功',
        type: 'success'
      });
    },
    error: function (e) {
      $.message({
        message: '处理失败！',
        type: 'error'
      });
    }
  })
}