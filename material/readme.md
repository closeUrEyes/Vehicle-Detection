<!DOCTYPE html>
<html lang='zh-CN'>
<link href="https://assets.gitee.com/assets/projects/application-e1f640e791e660016444044fbd36fca5.css" media="screen" rel="stylesheet" type="text/css" />
<head>
<meta charset='utf-8'>
<meta content='IE=edge' http-equiv='X-UA-Compatible'>
<meta content='码云,代码托管,git,Git@OSC,gitee.com,开源,项目管理,版本控制,开源代码,代码分享,项目协作,开源项目托管,免费代码托管,Git代码托管,Git托管服务' name='Keywords'>
<meta content='码云(gitee.com)是开源中国社区团队推出的基于Git的快速的、免费的、稳定的在线代码托管平台,不限制私有库和公有库数量' name='Description'>
<title>
readme.md · TinyMind/project-vehicle-detect - 码云 Gitee.com
</title>
<link href="https://assets.gitee.com/assets/favicon-950947d692935bf7e0b1629a69cd89ed.ico" rel="shortcut icon" type="image/vnd.microsoft.icon" />
<meta content='gitee.com/ai100/project-vehicle-detect git https://gitee.com/ai100/project-vehicle-detect.git' name='go-import'>
<link href="https://assets.gitee.com/assets/application-4d70879c5a8e16dc87cd740becda18eb.css" media="screen" rel="stylesheet" type="text/css" />
<script src="https://assets.gitee.com/assets/application-f9f860c1730e831ab6c77971b110c24b.js" type="text/javascript"></script>
<script src="https://assets.gitee.com/assets/lib/jquery.timeago.zh-CN-bcd91c2c27a815fa9a395595874b592b.js" type="text/javascript"></script>

<meta content="authenticity_token" name="csrf-param" />
<meta content="kbQxTeGXeOpXYYGEXvjBwuzYl2wXYPUyrsSqZog7ZyY=" name="csrf-token" />
<script type="text/javascript">
//<![CDATA[
window.gon = {};gon.locale="zh-CN";gon.http_clone="https://gitee.com/ai100/project-vehicle-detect.git";gon.user_project="ai100/project-vehicle-detect";gon.manage_branch="\u7ba1\u7406\u5206\u652f";gon.manage_tag="\u7ba1\u7406\u6807\u7b7e";gon.enterprise_id=1755200;gon.ref="master";
//]]>
</script>
<script src="//res.wx.qq.com/open/js/jweixin-1.2.0.js" type="text/javascript"></script>
<script>
  var title = document.title.replace(/( - Gitee| - 码云)$/, ''),
    imgUrl = '',
    imgUrlEl = document.querySelector('meta[itemprop=image]')
  if (imgUrlEl) {
    imgUrl = imgUrlEl.getAttribute('content')
  } else {
    imgUrl = "https://gitee.com//logo_4wechat.png"
  }
  wx.config({
    debug: false,
    appId: "wxff219d611a159737",
    timestamp: "1526116345",
    nonceStr: "d811aa2fb6a233b91698d815d31b8aeb",
    signature: "cc870d68b1735fd3faae64a5c9263ea71d94795a",
    jsApiList: [
      'onMenuShareTimeline',
      'onMenuShareAppMessage'
    ]
  });
  wx.ready(function () {
    wx.onMenuShareTimeline({
      title: title, // 分享标题
      link: "https://gitee.com/ai100/project-vehicle-detect/blob/master/readme.md", // 分享链接，该链接域名或路径必须与当前页面对应的公众号JS安全域名一致
      imgUrl: imgUrl // 分享图标
    });
    wx.onMenuShareAppMessage({
      title: title, // 分享标题
      link: "https://gitee.com/ai100/project-vehicle-detect/blob/master/readme.md", // 分享链接，该链接域名或路径必须与当前页面对应的公众号JS安全域名一致
      desc: document.querySelector('meta[name=Description]').getAttribute('content'),
      imgUrl: imgUrl // 分享图标
    });
  });
  wx.error(function(res){
    console.error('err', res)
  });
</script>

<script type='text/x-mathjax-config'>
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    displayMath: [["$$","$$"],["\\[","\\]"]],
    processEscapes: true,
    skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    ignoreClass: "container|files",
    processClass: "markdown-body"
  }
});
</script>
<script src="https://gitee.com/uploads/resources/MathJax-2.7.2/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<!--[if lt IE 10]>
<script>
    window.location.href = "/incompatible.html";
</script>
<![endif]-->
</head>

<body class='git-project lang-zh-CN'>
<script src="https://assets.gitee.com/assets/projects/app-65fe55923c8b503d563c589908127f12.js" type="text/javascript"></script>
<header class='common-header fixed noborder' id='git-header-nav'>
<div class='ui container'>
<div class='ui menu'>
<div class='item gitosc-logo'>
<a href="/"><img class='ui inline image' height='28' src='/logo.svg?20171024' width='95'>
</a></div>
<a href="/explore" class="item " title="开源软件">开源软件
</a><a href="/enterprises" class="item " title="企业版">企业版
<sup class='ui red label'>
特惠
</sup>
</a><a href="https://blog.gitee.com/" class="item" id="gitee-blog" target="_blank" title="博客">博客
</a><div class='dropdown item loading ui' id='my-gitee-dropdown'>
<a href="/closeUrEyes/events">我的码云</a>
<i class='dropdown icon'></i>
<div class='menu'>
<div class='header user-enterprises'>
<a href="/closeUrEyes/enterprises" class="pull-right" target="_blank">全部</a>
企业
<span class='count'></span>
</div>
<div class='disabled item tip user-enterprises'>
无企业
</div>
<div class='header user-groups'>
<a href="/closeUrEyes/groups" class="pull-right" target="_blank">全部</a>
组织
<span class='count'></span>
</div>
<div class='disabled item tip user-groups'>
无组织
</div>
<div class='header user-projects'>
<a href="/closeUrEyes/projects" class="pull-right" target="_blank">全部</a>
项目
<span class='count'></span>
</div>
<div class='disabled item tip user-projects'>
无项目
</div>
</div>
</div>
<div class='right menu userbar' id='git-nav-user-bar'>
<div class='item'>
<form accept-charset="UTF-8" action="/search" autocomplete="on" data-text-filter="搜索格式不正确" data-text-require="搜索关键字不能少于1个" id="navbar-search-form" method="get"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /></div>
<div class='ui mini fluid input'>
<input id="navbar-search-input" name="search" placeholder="搜索项目、代码片段..." type="text" value="" />
<input id="group_id" name="group_id" type="hidden" />
<input id="project_id" name="project_id" type="hidden" value="3020011" />
<input id="navbar-search-type" name="type" type="hidden" />
</div>
</form>


</div>
<div class='item ui dropdown empty' data-count-path='/notifications/unread_count' data-mark-notice-path='/notifications/mark' id='notice-dropdown'>
<a href="/notifications" class="remind-button"><i class='iconfont icon-remind'></i>
<div class='notice-count total'></div>
</a><div class='notice-dropdown-panel menu'>
<div class='notice-dropdown-panel-header'>
<div class='tab active' data-data-path='/notifications/notices?scope=referer' data-html-path='/notifications/referer' data-scope='referer'>
<div class='content'>
@ 我
<div class='notice-count referer'></div>
</div>
</div>
<div class='tab' data-data-path='/notifications/notices?scope=infos' data-html-path='/notifications/infos' data-scope='infos'>
<div class='content'>
通知
<div class='notice-count infos'></div>
</div>
</div>
<div class='tab' data-data-path='/notifications/notices?scope=messages' data-html-path='/notifications/messages' data-scope='messages'>
<div class='content'>
私信
<div class='notice-count messages'></div>
</div>
</div>
</div>
<div class='item notice-dropdown-panel-container'>
<div class='ui dimmer over active'>
<div class='ui loader'></div>
</div>
<div class='notice-list'></div>
<div class='notice-dropdown-panel-blank'>
暂没有新消息
</div>
</div>
<div class='notice-dropdown-panel-footer'>
<div class='action'>
<div class='side left'>
<a href="javascript: void(0);" class="mark-notices">全部标记为已读
</a></div>
<div class='side right'>
<a href="javascript: void(0);" class="load-all" target="_blank">查看全部
</a></div>
</div>
</div>
</div>
</div>

<div class='ui dropdown link item' id='git-nav-create'>
<i class='iconfont icon-add-thin'></i>
<div class='right menu'>
<a href="/projects/new" class="item"><i class='add square icon'></i>
新建项目
</a><a href="/closeUrEyes/codes/new" class="item"><i class='code icon'></i>
发布代码片段
</a><a href="/organizations/new" class="item"><i class='group icon'></i>
创建组织
</a><a href="/enterprises/new" class="item"><i class='icon iconfont icon-enterprise'></i>
开通企业版
</a><a href="/projects/oauth_github" class="item"><i class='github icon'></i>
从 GitHub 导入项目
</a><a href="/projects/csdn_code" class="item"><i class='iconfont icon-logo_csdn icon'></i>
CODE 一键迁移
</a></div>
</div>
<div class='ui dropdown item' id='git-nav-user'>
<a href="/closeUrEyes/events"><img avatar="亓呈文" class="ui avatar image closeUrEyes-avatar" />
</a><i class='dropdown icon'></i>
<div class='right menu'>
<a href="/closeUrEyes/events" class="item"><div class='mayun-icon my-ic-user-home my-ic-user-home-dims'></div>
个人主页
</a><a href="/profile" class="item"><div class='mayun-icon my-ic-edit my-ic-edit-dims'></div>
设置
</a><div class='divider'></div>
<a href="/gists" class="item"><div class='iconfont icon-snippet2'></div>
代码片段
</a><a href="http://git.mydoc.io/" class="item" target="_blank"><div class='mayun-icon my-ic-help my-ic-help-dims'></div>
帮助
</a><div class='divider'></div>
<a href="/logout" class="item destroy-user-session" data-method="delete" rel="nofollow"><div class='mayun-icon my-ic-exit my-ic-exit-dims'></div>
退出
</a></div>
</div>
<script>
  $('.destroy-user-session').click(function() {
    $.cookie('access_token', null, { path: '/' });
  })
</script>

</div>
</div>
</div>
</header>
<script>
  Gitee.initNavbar()
  Gitee.initRepoRemoteWay()
</script>

<!--[if lt IE 10]>
<script>
  window.location.href = "/incompatible.html"
</script>
<![endif]-->

<div class='fixed-notice-messages'>
<div class='ui container'>
<div class='flash-messages' id='messages-container'></div>
</div>
</div>
<script>
  (function() {
    $(function() {
      var $error_box, alertTip, notify_content, notify_options, setCookie, template;
  
      template = '<div data-notify="container" class="ui {0} message" role="alert">' + '<i data-notify="dismiss" class="close icon"></i>' + '<span data-notify="message">{2}</span>' + '</div>';
      notify_content = null;
      notify_options = {};
      alertTip = '';
      $error_box = $(".flash_error.flash_error_box");
      if (notify_options.type === 'error' && $error_box.length > 0 && !$.isEmptyObject(notify_content.message)) {
        if (notify_content.message === 'captcha_fail') {
          alertTip = "验证码不正确";
        } else if (notify_content.message === 'not_found_in_database') {
          alertTip = "帐号或者密码错误";
        } else if (notify_content.message === 'not_found_and_show_captcha') {
          alertTip = "帐号或者密码错误";
        } else {
          alertTip = notify_content.message;
        }
        $error_box.html(alertTip).show();
      } else if (notify_content) {
        notify_options.delay = 3000;
        notify_options.template = template;
        notify_options.offset = {
          x: 10,
          y: 30
        };
        notify_options.element = '#messages-container';
        $.notify(notify_content, notify_options);
      }
      setCookie = function(name, value) {
        $.cookie(name, value, {
          path: '/',
          expires: 365
        });
      };
      $('#remove-bulletin').on('click', function() {
        setCookie('remove_bulletin', "gitee-maintain-1516115940");
        $('#git-bulletin').hide();
      });
      return $('#remove-member-bulletin').on('click', function() {
        setCookie('remove_member_bulletin', "gitee_member_bulletin");
        $(this).parent().hide();
      });
    });
  
  }).call(this);
</script>

<div class='git-project-header'>
<div class='fixed-notice-messages'>
<div class='ui info icon floating message green' id='fetch-ok' style='display: none'>
<div class='content'>
<div class='header status-title'>
<i class='info icon status-icon'></i>
代码拉取完成，页面将自动刷新
</div>
</div>
</div>
<div class='ui info icon floating message error' id='fetch-error' style='display: none'>
<div class='content'>
<div class='header status-title'>
<i class='info icon status-icon'></i>
<span class='error_msg'></span>
</div>
</div>
</div>
</div>
<div class='ui container'>

<div class='git-project-header-details'>
<div class='git-project-header-actions'>
<span class='ui basic buttons'>
<a class='ui button donate' id='project-donate' title='捐赠'>
<i class='iconfont icon-donation'></i>
捐赠
</a>
<a class='ui button social-count' href='/ai100/project-vehicle-detect#project-donate-overview'>0</a>
<div class='ui small modal project-donate-modal' id='project-donate-modal'>
<i class='iconfont icon-close close'></i>
<div class='header'>项目捐赠</div>
<div class='content center aligned'>
项目的主人没有开启捐赠功能，快通知他让他开启吧！
<a id='send-message-to-author'>
&ensp;&ensp;
发送私信 &gt;&gt;
</a>
</div>
</div>
<div class='ui small modal wepay-qrcode'>
<i class='iconfont icon-close close'></i>
<div class='header'>
扫描微信二维码支付
<span class='wepay-cash'></span>
</div>
<div class='content weqcode-center'>
<img id='wepay-qrcode' src=''>
</div>
<div class='actions'>
<div class='ui cancel blank button'>取消</div>
<div class='ui ok orange button'>
支付完成
</div>
</div>
</div>
</span>

<span class='basic buttons ui watch-container'>
<a href="/ai100/project-vehicle-detect/unwatch" class="ui button unwatch" data-method="post" data-remote="true" rel="nofollow"><i class='iconfont icon-watch'></i>
Unwatch
</a><a href="/ai100/project-vehicle-detect/watch" class="ui button watch" data-method="post" data-remote="true" rel="nofollow"><i class='iconfont icon-watch'></i>
Watch
</a><a href="/ai100/project-vehicle-detect/watchers" class="ui button social-count" title="1">1
</a></span>
<span class='basic buttons star-container ui'>
<a href="/ai100/project-vehicle-detect/unstar" class="ui button unstar" data-method="post" data-remote="true" rel="nofollow"><i class='iconfont icon-star'></i>
Unstar
</a><a href="/ai100/project-vehicle-detect/star" class="ui button star" data-method="post" data-remote="true" rel="nofollow"><i class='iconfont icon-star'></i>
Star
</a><a href="/ai100/project-vehicle-detect/stargazers" class="ui button social-count" title="0">0
</a></span>
<span class='ui basic buttons fork-container' data-content=''>
<a href="#" class="ui button fork " id="fork-top-button"><i class='iconfont icon-fork'></i>
Fork
</a><a href="/ai100/project-vehicle-detect/members" class="ui button social-count" title="0">0
</a></span>
</div>
<h2 class='git-project-title'>
<a href='/ai100'><i class="iconfont icon-enterprise-badge" title="这是一个企业项目"></i></a> <i class="iconfont icon-project-public" title="这是一个公开项目"></i> <a href="/ai100" class="author" title="TinyMind">TinyMind</a> / <a href="/ai100/project-vehicle-detect" class="repository" style="padding-bottom: 0px" target="" title="project-vehicle-detect">project-vehicle-detect</a>

<input id="project_title" name="project_title" type="hidden" value="TinyMind/project-vehicle-detect" />
</h2>
</div>
</div>
<div class='row' id='import-result-message' style='padding-top: 0px; display: none'>
<div class='sixteen wide column'>
<div class='ui icon yellow message status-color'>
<i class='info icon status-icon' style='width:60px;padding-right:12px;'></i>
<i class='close icon'></i>
<div class='header status-title'>
同步状态
</div>
<span class='status-message'></span>
</div>
</div>
</div>
<div class='ui small modal' id='modal-fork-project'>
<i class='icon-close iconfont close'></i>
<div class='header'>
Fork 项目
</div>
<div class='content'>
<div class='fork-info-content'>
<div class='ui segment fork_project_loader'>
<div class='ui active inverted dimmer'>
<div class='ui text loader'>加载中</div>
</div>
</div>
</div>
</div>
<div class='actions fork-action hide'>
<a class='cancel'>&emsp;取消&emsp;</a>
<div class='ui large button orange ok'>&emsp;确认&emsp;</div>
</div>
</div>
<script>
  (function() {
    var $modalFork;
  
    this.title_project_path = 'project-vehicle-detect';
  
    this.title_fork_url = '/ai100/project-vehicle-detect/sync_fork';
  
    $modalFork = $('#modal-fork-project');
  
    $('#fork-top-button').on('click', function(e) {
      e.preventDefault();
      $modalFork.modal('show');
      return setTimeout(function() {
        return $.ajax({
          url: "/ai100/project-vehicle-detect/fork_project_info"
        });
      }, 500);
    });
  
    $('#modal-fork-project .close-button').on('click', function(e) {
      return $modalFork.modal('hide');
    });
  
  }).call(this);
</script>
<script>
  (function() {
    $('#modal-fork-project').modal({
      transition: 'fade'
    });
  
  }).call(this);
</script>
<style>
  i.loading {
    -webkit-animation: icon-loading 1.2s linear infinite;
    animation: icon-loading 1.2s linear infinite;
  }
  .qrcode_cs{
    float: left;
  }
</style>

<div class='git-project-nav'>
<div class='ui container'>
<div class='ui secondary pointing menu'>
<a href="/ai100/project-vehicle-detect" class="item active"><i class='iconfont icon-code'></i>
代码
</a><a href="/ai100/project-vehicle-detect/issues" class="item "><i class='iconfont icon-issue'></i>
Issues
<span class='ui mini circular label'>
0
</span>
</a><a href="/ai100/project-vehicle-detect/pulls" class="item "><i class='iconfont icon-pull-request'></i>
Pull Requests
<span class='ui mini circular label'>
0
</span>
</a><a href="/ai100/project-vehicle-detect/attach_files" class="item "><i class='iconfont icon-annex'></i>
附件
<span class='ui mini circular label'>0</span>
</a><a href="/ai100/project-vehicle-detect/wikis" class="item "><i class='iconfont icon-wiki'></i>
Wiki
<span class='ui mini circular label'>
0
</span>
</a><div class='item'>
<div class='ui dropdown git-project-service'>
<div class='text'>
<i class='iconfont icon-service'></i>
服务
</div>
<i class='dropdown icon'></i>
<div class='menu' style='display:none'>
<a href="/ai100/project-vehicle-detect/paas/huaweicloud_swr" class="item"><img alt="Huaweirqy" src="https://assets.gitee.com/assets/huaweirqy-c24d1adc35c0ecb2bc1e39f6e033835c.png" />
<div class='item-title'>
华为容器云
</div>
</a><a href="/ai100/project-vehicle-detect/paas/huaweicloud_cse" class="item"><img alt="Hauweiwfw" src="https://assets.gitee.com/assets/hauweiwfw-9edbcf60ca0a5d0de81d2fc804efa9ac.png" />
<div class='item-title'>
华为微服务平台
</div>
</a><a href="/ai100/project-vehicle-detect/paas/select_platform" class="item"><img alt="Mopaas_mini" src="https://assets.gitee.com/assets/mopaas_mini-72f0d5aeae31630ae89f624dfb0c23ca.png" />
<div class='item-title'>
MoPaaS V3
</div>
</a><a href="/ai100/project-vehicle-detect/paas/select_platform" class="item"><img alt="Baidu_mini" src="https://assets.gitee.com/assets/baidu_mini-3922d767f78ffc0fe5b0ad5ded300005.png" />
<div class='item-title'>
百度应用引擎
</div>
</a><a href="/ai100/project-vehicle-detect/quality_analyses?platform=sonar" class="item"><img alt="Sonar_mini" src="https://assets.gitee.com/assets/sonar_mini-6270e37950512a0bf0a05ac5b9b11243.png" />
<div class='item-title'>
代码质量分析
</div>
</a><a href="/ai100/project-vehicle-detect/quality_analyses?platform=codesafe" class="item"><img alt="Dmws" src="https://assets.gitee.com/assets/dmws-7fce33f3494048913a196a40f998a9ba.png" />
<div class='item-title'>
源代码缺陷检测
</div>
</a><a href="/ai100/project-vehicle-detect/pages" class="item"><img alt="Gitee" src="https://assets.gitee.com/assets/gitee-d6fb391be28450a587df71dda0325f60.png" />
<div class='item-title'>
Gitee Pages
</div>
</a></div>
</div>
</div>
</div>
</div>
</div>
<script>
  $('.git-project-nav .ui.dropdown').dropdown({ action: 'nothing' });
</script>
<style>
  .git-project-nav i.checkmark.icon {
    color: green;
  }
</style>
<script>
  $('#git-versions.dropdown').dropdown();
  $.ajax({
    url:"/ai100/project-vehicle-detect/access/add_access_log",
    type:"GET"
  });
</script>

</div>

<div class='git-project-content-wrapper'>
<div class='ui container'>
<div class='git-project-content' id='git-project-content'>
<div class='row' id='git-detail-clone'>
<div class='git-project-desc-wrapper'>
<div class='git-project-desc'>
<span class='git-project-desc-text'>
<i>暂无描述</i>
</span>
<span class='git-project-homepage'>
</span>
</div>
<div class='git-project-desc-edit ui form'>
<div class='fields'>
<div class='eight wide field'>
<div class='ui small input'>
<input name='project[description]' placeholder='描述' type='text' value=''>
</div>
</div>
<div class='four wide field'>
<div class='ui small input'>
<input data-regex-value='(^$)|(^(http|https):\/\/(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]).*)|(^(http|https):\/\/[a-zA-Z0-9]+([_\-\.]{1}[a-zA-Z0-9]+)*\.[a-zA-Z]{2,10}(:[0-9]{1,10})?(\/.*)?$)' name='project[homepage]' placeholder='项目主页(eg: https://gitee.com)' type='text'>
</div>
</div>
<button class='ui positive button btn-save'>
保存更改
</button>
<button class='ui button btn-cancel-edit'>
取消
</button>
</div>
</div>
<script>
  $(function () {
    new ProjectInfoEditor({
      el: '.git-project-desc-wrapper',
      homepage: null,
      description: "",
      url: "/ai100/project-vehicle-detect/update_description",
      modalHelper: Gitee.modalHelper
    })
  })
</script>
</div>

</div>
<div class='ui horizontal hollow nopadding segments git-project-stats'>
<div class='ui segment center aligned'>
<a href="/ai100/project-vehicle-detect/commits/master"><i class='iconfont icon-commit'></i>
<b id='commits_count'>2</b>
次提交
</a></div>
<div class='ui segment center aligned'>
<a href="/ai100/project-vehicle-detect/branches"><i class='iconfont icon-branches'></i>
<b id='branches_count'>1</b>
个分支
</a></div>
<div class='ui segment center aligned'>
<a href="/ai100/project-vehicle-detect/tags"><i class='iconfont icon-tag'></i>
<b id='tags_count'>0</b>
个标签
</a></div>
<div class='ui segment center aligned'>
<a href="/ai100/project-vehicle-detect/releases"><i class='iconfont icon-release'></i>
<b id='releases_count'>0</b>
个发行版
</a></div>
<div class='ui segment center aligned'>
<a href="/ai100/project-vehicle-detect/contributors?ref=master"><i class='iconfont icon-collaborators'></i>
<b id='collaborators_count'></b>
<span class='contributor_text'>
正在获取贡献者
</span>
</a></div>
</div>
<div class='git-project-bread' id='git-project-bread'>
<div class='ui right floated horizontal list'>
<div class='item'>
<div class='ui orange button' id='btn-dl-or-clone'>
克隆/下载
<i class='dropdown icon'></i>
</div>
</div>
</div>
<div class='ui horizontal list'>
<div class='item item git-project-branch-item'>
<input id="path" name="path" type="hidden" value="readme.md" />
<div class='ui top left pointing dropdown gradient button dropdown-has-tabs' id='git-project-branch'>
<input id="ref" name="ref" type="hidden" value="master" />
<div class='default text'>
master
</div>
<i class='dropdown icon'></i>
<div class='menu'>
<div class='ui left icon search input'>
<i class='iconfont icon-search'></i>
<input placeholder='搜索分支' type='text'>
</div>
<div class='tab-menu'>
<div class='tab-menu-action' data-tab='branches'>
<a href="/ai100/project-vehicle-detect/branches/recent" class="ui small basic blue button">管理</a>
</div>
<div class='tab-menu-action' data-tab='tags'>
<a href="/ai100/project-vehicle-detect/tags" class="ui small basic blue button">管理</a>
</div>
<div class='tab-menu-item' data-placeholder='搜索分支' data-tab='branches'>
分支 (1)
</div>
</div>
<div class='tab scrolling menu' data-tab='branches'>
<div class='item' data-value='master'>master</div>
</div>
</div>
</div>
</div>
<div class='item'>
<div class='repo-index repo-none-index' style='margin-left:0px;'>
<div class='ui horizontal list repo-action-list'>
<div class='item'>
<a href="/ai100/project-vehicle-detect/pull/new/ai100:master...ai100:master" class="ui gradient button repo-action left attached">+ Pull Request</a>
<a href="/ai100/project-vehicle-detect/issues/new" class="ui gradient button repo-action right attached">+ Issue</a>
</div>
<div class='item'>
<div class='ui pointing right top dropdown gradient button' id='git-project-file'>
<div class='text'>文件</div>
<i class='dropdown icon'></i>
<div class='menu'>
<div class='disabled item'>新建文件</div>
<div class='disabled item'>上传文件</div>
<a class='item repo-action' id='search-files'>
搜索文件
</a>
</div>
</div>
</div>
<div class='item'>
<a href="/ai100/project-vehicle-detect/widget" class="ui gradient button repo-action"><i class='iconfont icon-widget icon-orange'></i>
挂件
</a></div>
</div>
</div>
</div>
<div class='breadcrumb_path item' style='margin-left: 0; padding-left: 0'>
<div class='ui breadcrumb path' id='path-breadcrumb'>
<a href="/ai100/project-vehicle-detect/tree/master" class="section repo-name" data-direction="back" style="font-weight: bold">project-vehicle-detect
</a><div class='divider'>
/
</div>
<strong>
readme.md
</strong>
</div>

</div>
</div>
<div class='ui popup bottom right transition hidden git-project-download-panel'>
<div class='ui small secondary pointing menu'>
<a class='item active' data-type='http' data-url='https://gitee.com/ai100/project-vehicle-detect.git'>HTTPS</a>
<a class='item' data-type='ssh' data-url='git@gitee.com:ai100/project-vehicle-detect.git'>SSH</a>
</div>
<div class='ui fluid right labeled small input'>
<input id="project_clone_url" name="project_clone_url" onclick="focus();select()" readonly="readonly" type="text" value="https://gitee.com/ai100/project-vehicle-detect.git" />
<div class='ui basic label'>
<div class='ui small basic orange button' data-clipboard-target='#project_clone_url' id='btn-copy-clone-url'>
复制
</div>
</div>
</div>
<hr>
<a href="/ai100/project-vehicle-detect/repository/archive/master.zip" class="ui fluid tiny download link button"><i class='icon download'></i>
下载ZIP
</a><hr>
<a href='/enterprises?from=gea-dl-A'>
<div class='dl ent-poster'>
<div class='carousel slide' data-interval='5000' data-ride='carousel'>
<div class='carousel-inner'>
<div class='item img-1 active'>
<img src='/index/ent_poster/banner_5_1_a.png'>
</div>
<div class='item img-2'>
<img src='/index/ent_poster/banner_5_2_a.png'>
</div>
<div class='item img-3'>
<img src='/index/ent_poster/banner_5_3_a.png'>
</div>
</div>
</div>
</div>
</a>

</div>
</div>
<style>
  .ui.dropdown .menu>.header{text-transform:none}
</style>
<script>
  $(document).ready(function () {
    var $contri_count = $('#collaborators_count')
    var $contri_text = $('.contributor_text')
    $.ajax({
      url: '/ai100/project-vehicle-detect/contributors_count?ref=master',
      method: 'GET',
      success: function(data) {
        if (data.status === 1) {
          $contri_count.text(data.contributors_count);
          $contri_text.text('位贡献者')
        } else {
          $contri_text.text('获取失败')
        }
      }
    });
    var $tip = $('#apk-download-tip');
    if (!$tip.length) {
      return;
    }
    $tip.find('.btn-close').on('click', function () {
      $tip.slideUp();
    });
  });
  Gitee.initTabsInDropdown($('#git-project-branch').dropdown({
    fullTextSearch: true,
    onChange: function (value, text) {
      var path = $('#path').val();
      var href = ['/ai100/project-vehicle-detect/tree', value, path].join('/');
      window.location.href = href;
    }
  }));
  $('#git-project-file').dropdown({ action: 'hide' });
  (function(){
    function pathAutoRender() {
      var $parent = $('#git-project-bread'),
          $child = $('#git-project-bread').children('.ui.horizontal.list'),
          mainWidth = 0;
      $child.each(function (i,item) {
        mainWidth += $(item).width()
        });
      $('.breadcrumb.path.fork-path').remove();
      if (mainWidth > 995) {
        $('#path-breadcrumb').hide();
        $parent.append('<div class="ui breadcrumb path fork-path">' + $('#path-breadcrumb').html() + '<div/>')
      } else {
        $('#path-breadcrumb').show();
      }
    }
    window.pathAutoRender = pathAutoRender;
    pathAutoRender();
  })();
</script>
<script>
  (function() {
    var $btnCopy, $input, $item, $panel, clipboard, remoteWay;
  
    $input = $('#project_clone_url');
  
    remoteWay = 'http';
  
    clipboard = new Clipboard('#btn-copy-clone-url');
  
    $panel = $('.git-project-download-panel');
  
    $btnCopy = $('#btn-copy-clone-url');
  
    $panel.find('.menu > .item').on('click', function() {
      var $item;
  
      $item = $(this).addClass('active');
      $item.siblings().removeClass('active');
      $input.val($item.attr('data-url'));
      return setCookie('remote_way', $item.attr('data-type'), 360);
    });
  
    $('#btn-dl-or-clone').popup({
      popup: $panel,
      position: 'bottom right',
      on: 'click'
    });
  
    if (remoteWay) {
      $item = $panel.find('.item[data-type="' + remoteWay + '"]');
      if ($item.length === 0) {
        $item = $panel.find('.item[data-type="http"]');
      }
      $item.addClass('active').siblings().removeClass('active');
      $input.val($item.attr('data-url'));
    }
  
    clipboard.on('success', function() {
      $btnCopy.popup({
        content: '已复制',
        position: 'right center',
        onHidden: function() {
          return $btnCopy.popup('destroy');
        }
      });
      return $btnCopy.popup('show');
    });
  
    clipboard.on('error', function() {
      $btnCopy.popup({
        content: '复制失败，请手动复制',
        position: 'right center',
        onHidden: function() {
          return $btnCopy.popup('destroy');
        }
      });
      return $btnCopy.popup('show');
    });
  
  }).call(this);
</script>

<div class='row column tree-holder' id='tree-holder'>
<div class='tree-content-holder' id='tree-content-holder'>
<div class='file_holder'>
<div class='file_title'>
<div class='options'><div class='ui mini buttons basic'>
<textarea id="blob_raw" name="blob_raw" style="display:none;">
# 实战 车辆检测及型号识别&#x000A;&#x000A;## 简介&#x000A;车辆检测及型号识别广泛应用于物业，交通等的管理场景中。通过在停车场出入口，路口，高速卡口等位置采集的图片数据，对车辆的数量型号等进行识别，可以以较高的效率对车型，数量信息等进行采集。通过采集的数据，在不同的场景中可以辅助不同的业务开展。如商场停车位的规划，路况规划，或者公安系统追踪肇事车辆等等。&#x000A;&#x000A;在第七周的作业中，学员们已经掌握了使用slim框架来对植物进行分类识别。&#x000A;&#x000A;在第八周的作业中，学员们已经掌握使用slim物体检测框架来进行物体的检测和识别。&#x000A;&#x000A;本项目中，将会综合第七周作业内容和第八周的作业内容,实现一个车辆检测的工业级系统。&#x000A;&#x000A;## 作业内容&#x000A;&#x000A;学员需要利用tensorflow提供的slim图片分类框架和物体检测框架实现一个可以对任意图片进行车辆检测的系统。&#x000A;&#x000A;## 评价标准&#x000A;&#x000A;### 成果1, 一整套可以运行的系统&#x000A;包含代码和详细的文档。文档要求可操作。能够按照文档的描述搭建系统并运行。文档不全者不予及格。&#x000A;&#x000A;系统要求能检测任意图片并给出合理的输出。&#x000A;&#x000A;### 成果2, 提供一个演示视频&#x000A;视频内容：从任意图片网站上，随机下载一张有汽车在内的图片，送入系统进行检测。可以输出并显示图片中车辆的位置和型号等信息。没有车辆的图片可以给出没有检测到的提示。&#x000A;&#x000A;## 数据集&#x000A;&#x000A;本作业提供一个车辆分类的数据集。&#x000A;&#x000A;本作业提供的数据集分类参考数据集中的labels.txt文件：&#x000A;&#x000A;共48856张图片&#x000A;其43971张作为训练集，4885张作为验证集。&#x000A;&#x000A;数据已经预先打包成tfrecord格式，数据格式与第七周作业相同，第七周的代码可以直接使用。请联系课程管理人员获取训练数据。&#x000A;&#x000A;&#x000A;## 要点提示&#x000A;&#x000A;- 系统的输入输出不做要求，能够正常演示即可。&#x000A;    - 推荐的输入方式有：&#x000A;        - 命令行直接指定待识别文件&#x000A;        - 搭建一个web系统，使用表单方式上传文件&#x000A;        - 搭建一个native程序，使用pyqt等GUI框架搭建GUI界面&#x000A;    - 推荐的输出方式：&#x000A;        - 将检测结果写入文件&#x000A;        - 使用matplotlib显示检测结果&#x000A;        - 搭建一个web系统，在web页上显示结果&#x000A;        - 搭建一个native程序，使用pyqt等GUI框架搭建GUI界面&#x000A;- 训练数据集为分类数据，在1080Ti显卡上，以inceptionv4网络，0.001的学习率，利用google提供的预训练模型，在6～8个小时的训练后可以得到top1 80%的准确率。经过24个小时的训练后，top1可以达到88%。&#x000A;</textarea>
<a href="#" class="ui button" id="copy-text" style="border-left: none;">一键复制</a>
<a href="/ai100/project-vehicle-detect/edit/master/readme.md" class="ui button disabled has_tooltip edit-blob" title="无编辑权限">编辑</a>
<a href="/ai100/project-vehicle-detect/raw/master/readme.md" class="ui button edit-raw">原始数据</a>
<a href="/ai100/project-vehicle-detect/blame/master/readme.md" class="ui button edit-blame">按行查看</a>
<a href="/ai100/project-vehicle-detect/commits/master/readme.md" class="ui button edit-history">历史</a>
</div>
<script>
  try {
    if((gon.wait_fork!=undefined && gon.wait_fork==true) || (gon.wait_fetch!=undefined && gon.wait_fetch==true)){
      $('.edit-blob').popup({content:"当前项目正在后台处理中,暂时无法编辑", on: 'hover', delay: { show: 200, hide: 200 }});
      $('.edit-blob').click(function(e){
        e.preventDefault();
      })
    }
  
    setUrl = function(){
      var params = window.location.search
      if (params==undefined || $.trim(params).length==0) return;
      $('span.options').children('.basic').find('a').each(function(index,ele){
        var origin_href = $(ele).attr('href');
        if (origin_href!="#" && origin_href.indexOf('?') == -1){
          $(ele).attr('href',origin_href+params);
        }
      });
    }
  
    setUrl();
  
  
    var $btncopy = $("#copy-text");
    $(document).ready(function() {
      $btncopy.popup({
        content: isCopied ? '已复制':'点击复制'
      })
    })
    var isCopied = false;
    $btncopy.on('click', function() {
      try {
        var clipboard = new Clipboard("#copy-text", {
          text: function(trigger) {
            return $("#blob_raw").val();
          }
        });
        var success = document.execCommand('copy');
        if (success) {
          $btncopy.trigger('copied', ['已复制']);
          isCopied = true;
        } else {
          alert("复制失败，请手动复制")
        }
      } catch (err) {
        $btncopy.trigger('copied', ['点击复制']);
      }
    });
  
    $btncopy.bind('copied', function(event, message) {
      $(this).popup('hide');
      $(this).popup('destroy');
      isCopied = true;
      $(this).popup({content: isCopied ? '已复制':'点击复制'})
      $(this).popup('show');
    });
  } catch (error) {
    console.log('blob/action error:' + error);
  }
</script>
</div>
<div class='blob-description'>
<i class='iconfont icon-readme'></i>
<span class='file_name'>
readme.md
<small>2.67 KB</small>
</span>
→
<span class='recent-commit' style='margin-top: 0.7rem'>
<a class="commit-author-link" href="mailto:dwSun@Capsule"><img alt="" avatar="dwSun" class="mini avatar circular ui image " /> <span class="commit-author-name">dwSun</span></a>
<span>提交于</span>
<span class='timeago commit-date' title='2018-01-24 14:59:00 +0800'>
2018-01-24 14:59
</span>
.
<a href="/ai100/project-vehicle-detect/commit/c4e6dbf89f0f30468d44cfba4990a52916960b89" title="add description">add description</a>
</span>

</div>
</div>
<div class='clearfix'></div>
<div class='file_content markdown-body'>
<h1>&#x000A;<a id="实战-车辆检测及型号识别" class="anchor" href="#%E5%AE%9E%E6%88%98-%E8%BD%A6%E8%BE%86%E6%A3%80%E6%B5%8B%E5%8F%8A%E5%9E%8B%E5%8F%B7%E8%AF%86%E5%88%AB"></a>实战 车辆检测及型号识别</h1>&#x000A;<h2>&#x000A;<a id="简介" class="anchor" href="#%E7%AE%80%E4%BB%8B"></a>简介</h2>&#x000A;<p>车辆检测及型号识别广泛应用于物业，交通等的管理场景中。通过在停车场出入口，路口，高速卡口等位置采集的图片数据，对车辆的数量型号等进行识别，可以以较高的效率对车型，数量信息等进行采集。通过采集的数据，在不同的场景中可以辅助不同的业务开展。如商场停车位的规划，路况规划，或者公安系统追踪肇事车辆等等。</p>&#x000A;<p>在第七周的作业中，学员们已经掌握了使用slim框架来对植物进行分类识别。</p>&#x000A;<p>在第八周的作业中，学员们已经掌握使用slim物体检测框架来进行物体的检测和识别。</p>&#x000A;<p>本项目中，将会综合第七周作业内容和第八周的作业内容,实现一个车辆检测的工业级系统。</p>&#x000A;<h2>&#x000A;<a id="作业内容" class="anchor" href="#%E4%BD%9C%E4%B8%9A%E5%86%85%E5%AE%B9"></a>作业内容</h2>&#x000A;<p>学员需要利用tensorflow提供的slim图片分类框架和物体检测框架实现一个可以对任意图片进行车辆检测的系统。</p>&#x000A;<h2>&#x000A;<a id="评价标准" class="anchor" href="#%E8%AF%84%E4%BB%B7%E6%A0%87%E5%87%86"></a>评价标准</h2>&#x000A;<h3>&#x000A;<a id="成果1-一整套可以运行的系统" class="anchor" href="#%E6%88%90%E6%9E%9C1-%E4%B8%80%E6%95%B4%E5%A5%97%E5%8F%AF%E4%BB%A5%E8%BF%90%E8%A1%8C%E7%9A%84%E7%B3%BB%E7%BB%9F"></a>成果1, 一整套可以运行的系统</h3>&#x000A;<p>包含代码和详细的文档。文档要求可操作。能够按照文档的描述搭建系统并运行。文档不全者不予及格。</p>&#x000A;<p>系统要求能检测任意图片并给出合理的输出。</p>&#x000A;<h3>&#x000A;<a id="成果2-提供一个演示视频" class="anchor" href="#%E6%88%90%E6%9E%9C2-%E6%8F%90%E4%BE%9B%E4%B8%80%E4%B8%AA%E6%BC%94%E7%A4%BA%E8%A7%86%E9%A2%91"></a>成果2, 提供一个演示视频</h3>&#x000A;<p>视频内容：从任意图片网站上，随机下载一张有汽车在内的图片，送入系统进行检测。可以输出并显示图片中车辆的位置和型号等信息。没有车辆的图片可以给出没有检测到的提示。</p>&#x000A;<h2>&#x000A;<a id="数据集" class="anchor" href="#%E6%95%B0%E6%8D%AE%E9%9B%86"></a>数据集</h2>&#x000A;<p>本作业提供一个车辆分类的数据集。</p>&#x000A;<p>本作业提供的数据集分类参考数据集中的labels.txt文件：</p>&#x000A;<p>共48856张图片&#x000A;其43971张作为训练集，4885张作为验证集。</p>&#x000A;<p>数据已经预先打包成tfrecord格式，数据格式与第七周作业相同，第七周的代码可以直接使用。请联系课程管理人员获取训练数据。</p>&#x000A;<h2>&#x000A;<a id="要点提示" class="anchor" href="#%E8%A6%81%E7%82%B9%E6%8F%90%E7%A4%BA"></a>要点提示</h2>&#x000A;<ul class="task-list">&#x000A;<li>系统的输入输出不做要求，能够正常演示即可。&#x000A;<ul class="task-list">&#x000A;<li>推荐的输入方式有：&#x000A;<ul class="task-list">&#x000A;<li>命令行直接指定待识别文件</li>&#x000A;<li>搭建一个web系统，使用表单方式上传文件</li>&#x000A;<li>搭建一个native程序，使用pyqt等GUI框架搭建GUI界面</li>&#x000A;</ul>&#x000A;</li>&#x000A;<li>推荐的输出方式：&#x000A;<ul class="task-list">&#x000A;<li>将检测结果写入文件</li>&#x000A;<li>使用matplotlib显示检测结果</li>&#x000A;<li>搭建一个web系统，在web页上显示结果</li>&#x000A;<li>搭建一个native程序，使用pyqt等GUI框架搭建GUI界面</li>&#x000A;</ul>&#x000A;</li>&#x000A;</ul>&#x000A;</li>&#x000A;<li>训练数据集为分类数据，在1080Ti显卡上，以inceptionv4网络，0.001的学习率，利用google提供的预训练模型，在6～8个小时的训练后可以得到top1 80%的准确率。经过24个小时的训练后，top1可以达到88%。</li>&#x000A;</ul></div>
<script>
  (function() {
    $('.file_content.markdown-body pre').each(function(i, block) {
      return hljs.highlightBlock(block);
    });
  
  }).call(this);
</script>

</div>
</div>
<div class='tree_progress'></div>
<div class='ui small modal' id='modal-linejump'>
<div class='ui custom form content'>
<div class='field'>
<div class='ui right action input'>
<input placeholder='跳转至某一行...' type='number'>
<div class='ui orange button'>
跳转
</div>
</div>
</div>
</div>
</div>

<div class='row column inner-comment' id='blob-comment'>
<input id="comment_path" name="comment_path" type="hidden" value="readme.md" />
<div class='tree-comments'>
<h3 id='tree_comm_title'>
<i class='iconfont icon-comment'></i>
评论
(
<span class='comments-count'>
0
</span>
)
</h3>
<div class='ui threaded comments middle aligned' id='notes-list'></div>
<input id="ajax_add_note_id" name="ajax_add_note_id" type="hidden" />
<div class='text-center'>
<div class='tip-loading' style='display: none'>
<div class='ui active mini inline loader'></div>
正在加载...
</div>
</div>
</div>
<script>
  (function() {
    var $btnLoad, $comments, $container, $tipLoading, checkLoad, commentsCount, err, loadComments, page, path;
  
    page = 1;
  
    commentsCount = 0;
  
    $container = $('.tree-comments');
  
    $comments = $container.find('.ui.comments');
  
    $tipLoading = $container.find('.tip-loading');
  
    $btnLoad = $container.find('.btn-load-more');
  
    if (commentsCount < 1) {
      return;
    }
  
    if ($('#comment_path').val() === '') {
      path = '/';
    } else {
      path = $('#comment_path').val();
    }
  
    loadComments = function() {
      $btnLoad.hide();
      $tipLoading.show();
      return $.ajax({
        url: '/ai100/project-vehicle-detect/comment_list',
        data: {
          page: page,
          path: path
        },
        success: function(data) {
          var err;
  
          try {
            $tipLoading.hide();
            $btnLoad.show();
            if (data.status !== 0) {
              $btnLoad.text('无更多评论');
              return $btnLoad.prop('disabled', true).addClass('disabled');
            } else {
              TreeComment.CommentListHandler(data);
              page += 1;
              if (data.comments_count < 10) {
                $('#ajax_add_note_id').val('');
                $btnLoad.text('无更多评论');
                $btnLoad.prop('disabled', true).addClass('disabled');
              }
              $comments.find('.timeago').timeago();
              $comments.find('.commenter-role-label').popup();
              noteAnchorLocater.setup();
              toMathMlCode('', 'comments');
              return $('.markdown-body pre code').each(function(i, block) {
                return hljs.highlightBlock(block);
              });
            }
          } catch (_error) {
            err = _error;
            return console.log('loadComments error:' + err);
          }
        }
      });
    };
  
    checkLoad = function() {
      var listTop, top;
  
      top = $(window).scrollTop();
      listTop = $container.offset().top;
      if (listTop >= top && listTop < top + $(window).height()) {
        $(window).off('scroll', checkLoad);
        return loadComments();
      }
    };
  
    $btnLoad.on('click', loadComments);
  
    loadComments();
  
    try {
      this.noteAnchorLocater.init({
        lastAnchorElm: '#tree_comm_title'
      });
    } catch (_error) {
      err = _error;
      console.log('noteAnchorLocater err:' + err);
    }
  
  }).call(this);
</script>

</div>
<div class='inner-comment-box' id='comment-box'>
<div class='tree-comment-form' id='tree-comment-box'>
<img avatar="亓呈文" class="ui avatar image" />
<div class='ui form'>
<input id="comment_url" name="comment_url" type="hidden" value="/ai100/project-vehicle-detect/comment" />
<div class='field'>
<textarea class='md-input' id='editor_comment' placeholder='在此输入你对项目想说的话...'></textarea>
</div>
<div class='ui field right aligned except-editor-textarea'>

<div class='ui orange button js-comment-button disabled' id='only_comment'>
评论
</div>
<div class='ui basic button js-close-discussion-note-form' id='comment_cancel_btn'>
取消
</div>
</div>
</div>
</div>
<script>
  (function() {
    $(function() {
      return TreeComment.init();
    });
  
  }).call(this);
</script>

</div>

<script>
  $('.ui.checkbox').checkbox('set unchecked')
</script>

</div>
</div>
</div>
</div>
<script>
  (function() {
    $(function() {
      Tree.init();
      return TreeCommentActions.init();
    });
  
  }).call(this);
</script>

<script>
  (function() {
    var donateModal;
  
    Gitee.modalHelper = new GiteeModalHelper({
      alertText: '提示',
      okText: '确定'
    });
  
    donateModal = new ProjectDonateModal({
      el: '#project-donate-modal',
      alipayUrl: '/ai100/project-vehicle-detect/alipay',
      wepayUrl: '/ai100/project-vehicle-detect/wepay',
      modalHelper: Gitee.modalHelper
    });
  
    if ("" === 'true') {
      donateModal.show();
    }
  
    $('#project-donate').on('click', function() {
      return donateModal.show();
    });
  
    $('#send-message-to-author').on('click', function() {
      donateModal.hide();
      return $('#project-user-message').trigger('click');
    });
  
  }).call(this);
</script>
<script>
  Tree.initHighlightTheme('white')
</script>

<script>
  $(function() {
    GitLab.GfmAutoComplete.dataSource = "/ai100/project-vehicle-detect/autocomplete_sources"
    GitLab.GfmAutoComplete.Emoji.assetBase = 'https://assets.gitee.com/assets/emoji'
    GitLab.GfmAutoComplete.setup();
  });
</script>

<footer id='git-footer-main'>
<div class='ui container'>
<div class='ui two column grid'>
<div class='column'>
<p><a href="https://gitee.com/" target="_blank">© Gitee.com  </a></p>
<div class='ui three column grid' id='footer-left'>
<div class='column'>
<div class='ui link list'>
<div class='item'>
<a href="/about_us" class="item">关于我们</a>
</div>
<div class='item'>
<a href="/terms" class="item">使用条款</a>
</div>
<div class='item'>
<a href="http://git.mydoc.io" class="item">帮助文档</a>
</div>
</div>
</div>
<div class='column'>
<div class='ui link list'>
<div class='item'>
<a href="/api/v5/swagger" class="item">OpenAPI</a>
</div>
<div class='item'>
<a href="/gitee-stars" class="item">码云封面人物</a>
</div>
<div class='item'>
<a href="/appclient" class="item">APP与插件下载</a>
</div>
</div>
</div>
<div class='column'>
<div class='ui link list'>
<div class='item'>
<a href="/links.html" class="item">合作伙伴</a>
</div>
<div class='item'>
<a href="https://gitee.com/git-osc/" class="item">更新日志</a>
</div>
<div class='item'>
<a href="/oschina/git-osc/issues" class="item">意见与建议</a>
</div>
<div class='item'>
<a href="/gists" class="item">代码片段</a>
</div>
</div>
</div>
</div>
</div>
<div class='column right aligned followus'>
<div class='qrcode weixin'>
<img alt="Qrcode-weixin" src="https://assets.gitee.com/assets/qrcode-weixin-8ab7378f5545710bdb3ad5c9d17fedfe.jpg" />
<p class='weixin-text'>微信公众号</p>
</div>
<div class='phone-and-qq column'>
<div class='ui list'>
<div class='item'>
400-898-2008 转2
</div>
<div class='item' id='git-footer-email'>
git#oschina.cn
</div>
<div class='item qq-and-weibo'>
<a href="//shang.qq.com/wpa/qunwpa?idkey=0d6c2fc0b5b71ac33405dd575bb490bf1a50e3c9a9f694e8a689cb59ee7dacc3" class="icon-popup" title="点击加入码云官方三群"><i class='iconfont icon-logo-qq'></i>
<span>三群：655903986</span>
</a><a href="http://weibo.com/mayunOSC" class="icon-popup" target="_blank" title="关注码云微博"><i class='iconfont icon-logo-weibo'></i>
<span>码云Gitee</span>
</a></div>
</div>
</div>
</div>
</div>
</div>
<div class='bottombar'>
<div class='ui container'>
<div class='ui grid'>
<div class='five wide column partner'>
本站带宽由 <a href="https://www.anchnet.com/" target="_blank" title="anchnet"><img alt="51idc" src="/51idc.png" /></a> 提供
</div>
<div class='eleven wide column right aligned'>
<div class='copyright'>
<a href="http://www.miitbeian.gov.cn/">粤ICP备12009483号-8</a>
深圳市奥思网络科技有限公司版权所有
</div>
<i class='icon world'></i>
<a href="/language/zh-CN">简 体
</a>/
<a href="/language/zh-TW">繁 體
</a>/
<a href="/language/en">English
</a></div>
</div>
</div>
</div>
</footer>
<script>
  Gitee.initFooter()
  $('#git-footer-main .icon-popup').popup({ position: 'bottom center' })
</script>


<div class='side-toolbar'>
<div class='button share-link'>
<i class='iconfont icon-share'></i>
</div>
<div class='ui popup'>
<div class='header'>
分享到
</div>
<div class='bdsharebuttonbox'>
<a class='item bds_weixin' data-cmd='weixin' title='分享到微信'>weixin</a>
<a class='item bds_tsina' data-cmd='tsina' title='分享到新浪微博'>sina</a>
<a class='item bds_sqq' data-cmd='sqq' title='分享到QQ好友'>qq</a>
<a class='item bds_qzone' data-cmd='qzone' title='分享到QQ空间'>qzone</a>
</div>
</div>
<div class='popup button' data-content='给项目拥有者发送私信' id='project-user-message'>
<i class='iconfont icon-message'></i>
</div>
<div class='popup button' data-content='评论' id='home-comment'>
<i class='iconfont icon-comment'></i>
</div>
<div class='button gotop popup' data-content='回到顶部' id='gotop'>
<i class='iconfont icon-top'></i>
</div>
</div>
<div class='ui modal tiny form' id='send-message-modal'>
<i class='iconfont icon-close close'></i>
<div class='header'>发送私信</div>
<div class='content'>
<div class='ui message hide'></div>
<div class='field'>
<textarea class='content-input' maxlength='255' placeholder='请在这里输入内容'></textarea>
</div>
</div>
<div class='actions'>
<div class='ui button blank cancel'>取消</div>
<div class='ui orange icon button disabled ok'>发送</div>
</div>
</div>
<script>
  var $mountedElem = $('#project-user-message'),
      $messageModal = $('#send-message-modal'),
      $modalTips = $messageModal.find('.message'),
      $contentInput = $messageModal.find('.content-input'),
      $captchaImage = $messageModal.find('.captcha_img'),
      $captchaInput = $messageModal.find('.captcha-field input'),
      $sendMessageBtn = $messageModal.find('.ok.button'),
      messageSending = false
  
  $mountedElem.on('click', function() {
    $captchaImage.trigger('click')
    $messageModal.modal('show')
  })
  
  $messageModal.modal({
    onApprove: function() {
      sendMessage()
      return false
    },
    onHidden: function() {
      $modalTips.hide()
    }
  })
  
  $captchaImage.on('click', function() {
    $captchaInput.val('')
  })
  
  $contentInput.on('change keydown', function(e) {
    var content = $(this).val()
    if ($.trim(content).length == 0) {
      $sendMessageBtn.addClass('disabled')
      return
    }
    $sendMessageBtn.removeClass('disabled')
    if ((e.ctrlKey || e.metaKey) && e.keyCode == 13) {
      $sendMessageBtn.trigger('click')
    }
  })
  
  function sendMessage() {
    if (messageSending) return
    $.ajax({
      url: '/notifications/messages',
      data: {
        content: $contentInput.val(),
        receiver_id: '1696555',
        receiver_username: '',
        captcha: $captchaInput.val()
      },
      type: 'POST',
      dataType: 'JSON',
      beforeSend: function() {
        setSendStatus(true)
      },
      success: function(res) {
        if (res.status != 200) {
          showTips(res.message || '可能由于网络原因，留言发送失败，请稍后再试', 'error')
        } else {
          $contentInput.val('')
          showTips('私信发送成功')
          setTimeout(function() { $messageModal.modal('hide') }, 500)
        }
        setSendStatus(false)
      },
      error: function(err) {
        showTips('可能由于网络原因，留言发送失败，请稍后再试', 'error')
        setSendStatus(false)
      }
    })
  }
  
  function showTips(text, type) {
    $modalTips.text(text).show()
    if (type == 'error') {
      $modalTips.removeClass('success').addClass('error')
    } else {
      $modalTips.removeClass('error').addClass('success')
    }
  }
  
  function setSendStatus(value) {
    messageSending = value
    if (messageSending) {
      $sendMessageBtn.addClass('loading')
      $contentInput.attr('readonly', true)
    } else {
      $sendMessageBtn.removeClass('loading')
      $contentInput.attr('readonly', false)
    }
  }
</script>

<script>
  (function() {
    $('#project-user-message').popup({
      position: 'left center'
    });
  
  }).call(this);
</script>
<script>
  Gitee.initSideToolbar({
    hasComment: true,
    commentUrl: '/ai100/project-vehicle-detect#project_comm_title'
  })
</script>
<script>
  window._bd_share_config={"common":{"bdSnsKey":{},"bdText":"分享到新浪微博","bdMini":"1","bdMiniList":["bdxc","tqf","douban","bdhome","sqq","thx","ibaidu","meilishuo","mogujie","diandian","huaban","duitang","hx","fx","youdao","sdo","qingbiji","people","xinhua","mail","isohu","yaolan","wealink","ty","iguba","fbook","twi","linkedin","h163","evernotecn","copy","print"],"bdPic":"","bdStyle":"1","bdSize":"32"},"share":{}};
</script>
<script src="/bd_share/static/api/js/share.js" type="text/javascript"></script>


<style>
  .float-left-box{display:none;position:fixed;left:0;bottom:0;z-index:99}.float-left-box .close-left{position:absolute;top:20px;left:25px;cursor:pointer}.float-left-box .float-people{width:200px;padding:10px}
</style>
<div class='float-left-box'>
<a href='/gitee-stars/9' target='_blank'>
<img alt="9_float_left_people" class="float-people" src="https://assets.gitee.com/assets/gitee_stars/9_float_left_people-00b5fd6a643934934d6b63d7c2359269.png" />
<img alt="9_float_left_close" class="close-left" src="https://assets.gitee.com/assets/gitee_stars/9_float_left_close-7877f8b6672f35b75c0dad8183bb7800.png" />
</a>
</div>
<script>
  var giteeStarsWidget = $('.float-left-box')
  if ($.cookie('visit-gitee-9') == 1) {
    giteeStarsWidget.hide()
  } else {
    giteeStarsWidget.show()
  }
  $('.close-left').on('click', function(e) {
    e.preventDefault()
    $.cookie('visit-gitee-9', 1, { path: '/', expires: 30})
    giteeStarsWidget.hide()
  })
</script>

<script>
  (function() {
    this.__gac = {
      domain: 'www.oschina.net'
    };
  
  }).call(this);
</script>
<script defer src='//www.oschina.net/public/javascripts/cjl/ga.js?t=20160926' type='text/javascript'></script>

</body>
</html>
