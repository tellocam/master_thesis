var _JUPYTERLAB;(()=>{"use strict";var e,r,t,n,a,o,i,l,u,s,f,p,d,c,h,v,b,g,m,y={211:(e,r,t)=>{var n={"./index":()=>t.e(296).then((()=>()=>t(296))),"./extension":()=>t.e(296).then((()=>()=>t(296)))},a=(e,r)=>(t.R=r,r=t.o(n,e)?n[e]():Promise.resolve().then((()=>{throw new Error('Module "'+e+'" does not exist in container.')})),t.R=void 0,r),o=(e,r)=>{if(t.S){var n="default",a=t.S[n];if(a&&a!==e)throw new Error("Container initialization failed as it has already been initialized with a different share scope");return t.S[n]=e,t.I(n,r)}};t.d(r,{get:()=>a,init:()=>o})}},w={};function j(e){var r=w[e];if(void 0!==r)return r.exports;var t=w[e]={id:e,exports:{}};return y[e](t,t.exports,j),t.exports}j.m=y,j.c=w,j.n=e=>{var r=e&&e.__esModule?()=>e.default:()=>e;return j.d(r,{a:r}),r},j.d=(e,r)=>{for(var t in r)j.o(r,t)&&!j.o(e,t)&&Object.defineProperty(e,t,{enumerable:!0,get:r[t]})},j.f={},j.e=e=>Promise.all(Object.keys(j.f).reduce(((r,t)=>(j.f[t](e,r),r)),[])),j.u=e=>e+".87b6a53d1d5b3f17143e.js?v=87b6a53d1d5b3f17143e",j.g=function(){if("object"==typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"==typeof window)return window}}(),j.o=(e,r)=>Object.prototype.hasOwnProperty.call(e,r),e={},r="ipyparallel-labextension:",j.l=(t,n,a,o)=>{if(e[t])e[t].push(n);else{var i,l;if(void 0!==a)for(var u=document.getElementsByTagName("script"),s=0;s<u.length;s++){var f=u[s];if(f.getAttribute("src")==t||f.getAttribute("data-webpack")==r+a){i=f;break}}i||(l=!0,(i=document.createElement("script")).charset="utf-8",i.timeout=120,j.nc&&i.setAttribute("nonce",j.nc),i.setAttribute("data-webpack",r+a),i.src=t),e[t]=[n];var p=(r,n)=>{i.onerror=i.onload=null,clearTimeout(d);var a=e[t];if(delete e[t],i.parentNode&&i.parentNode.removeChild(i),a&&a.forEach((e=>e(n))),r)return r(n)},d=setTimeout(p.bind(null,void 0,{type:"timeout",target:i}),12e4);i.onerror=p.bind(null,i.onerror),i.onload=p.bind(null,i.onload),l&&document.head.appendChild(i)}},j.r=e=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},(()=>{j.S={};var e={},r={};j.I=(t,n)=>{n||(n=[]);var a=r[t];if(a||(a=r[t]={}),!(n.indexOf(a)>=0)){if(n.push(a),e[t])return e[t];j.o(j.S,t)||(j.S[t]={});var o=j.S[t],i="ipyparallel-labextension",l=[];return"default"===t&&((e,r,t,n)=>{var a=o[e]=o[e]||{},l=a[r];(!l||!l.loaded&&(1!=!l.eager?n:i>l.from))&&(a[r]={get:()=>j.e(296).then((()=>()=>j(296))),from:i,eager:!1})})("ipyparallel-labextension","9.0.0"),e[t]=l.length?Promise.all(l).then((()=>e[t]=1)):1}}})(),(()=>{var e;j.g.importScripts&&(e=j.g.location+"");var r=j.g.document;if(!e&&r&&(r.currentScript&&"SCRIPT"===r.currentScript.tagName.toUpperCase()&&(e=r.currentScript.src),!e)){var t=r.getElementsByTagName("script");if(t.length)for(var n=t.length-1;n>-1&&(!e||!/^http(s?):/.test(e));)e=t[n--].src}if(!e)throw new Error("Automatic publicPath is not supported in this browser");e=e.replace(/#.*$/,"").replace(/\?.*$/,"").replace(/\/[^\/]+$/,"/"),j.p=e})(),t=e=>{var r=e=>e.split(".").map((e=>+e==e?+e:e)),t=/^([^-+]+)?(?:-([^+]+))?(?:\+(.+))?$/.exec(e),n=t[1]?r(t[1]):[];return t[2]&&(n.length++,n.push.apply(n,r(t[2]))),t[3]&&(n.push([]),n.push.apply(n,r(t[3]))),n},n=(e,r)=>{e=t(e),r=t(r);for(var n=0;;){if(n>=e.length)return n<r.length&&"u"!=(typeof r[n])[0];var a=e[n],o=(typeof a)[0];if(n>=r.length)return"u"==o;var i=r[n],l=(typeof i)[0];if(o!=l)return"o"==o&&"n"==l||"s"==l||"u"==o;if("o"!=o&&"u"!=o&&a!=i)return a<i;n++}},a=e=>{var r=e[0],t="";if(1===e.length)return"*";if(r+.5){t+=0==r?">=":-1==r?"<":1==r?"^":2==r?"~":r>0?"=":"!=";for(var n=1,o=1;o<e.length;o++)n--,t+="u"==(typeof(l=e[o]))[0]?"-":(n>0?".":"")+(n=2,l);return t}var i=[];for(o=1;o<e.length;o++){var l=e[o];i.push(0===l?"not("+u()+")":1===l?"("+u()+" || "+u()+")":2===l?i.pop()+" "+i.pop():a(l))}return u();function u(){return i.pop().replace(/^\((.+)\)$/,"$1")}},o=(e,r)=>{if(0 in e){r=t(r);var n=e[0],a=n<0;a&&(n=-n-1);for(var i=0,l=1,u=!0;;l++,i++){var s,f,p=l<e.length?(typeof e[l])[0]:"";if(i>=r.length||"o"==(f=(typeof(s=r[i]))[0]))return!u||("u"==p?l>n&&!a:""==p!=a);if("u"==f){if(!u||"u"!=p)return!1}else if(u)if(p==f)if(l<=n){if(s!=e[l])return!1}else{if(a?s>e[l]:s<e[l])return!1;s!=e[l]&&(u=!1)}else if("s"!=p&&"n"!=p){if(a||l<=n)return!1;u=!1,l--}else{if(l<=n||f<p!=a)return!1;u=!1}else"s"!=p&&"n"!=p&&(u=!1,l--)}}var d=[],c=d.pop.bind(d);for(i=1;i<e.length;i++){var h=e[i];d.push(1==h?c()|c():2==h?c()&c():h?o(h,r):!c())}return!!c()},i=(e,r)=>e&&j.o(e,r),l=e=>(e.loaded=1,e.get()),u=e=>Object.keys(e).reduce(((r,t)=>(e[t].eager&&(r[t]=e[t]),r)),{}),s=(e,r,t)=>{var a=t?u(e[r]):e[r];return Object.keys(a).reduce(((e,r)=>!e||!a[e].loaded&&n(e,r)?r:e),0)},f=(e,r,t,n)=>"Unsatisfied version "+t+" from "+(t&&e[r][t].from)+" of shared singleton module "+r+" (required "+a(n)+")",p=e=>{throw new Error(e)},d=e=>{"undefined"!=typeof console&&console.warn&&console.warn(e)},c=(e,r,t)=>t?t():((e,r)=>p("Shared module "+r+" doesn't exist in shared scope "+e))(e,r),h=(e=>function(r,t,n,a,o){var i=j.I(r);return i&&i.then&&!n?i.then(e.bind(e,r,j.S[r],t,!1,a,o)):e(r,j.S[r],t,n,a,o)})(((e,r,t,n,a,u)=>{if(!i(r,t))return c(e,t,u);var p=s(r,t,n);return o(a,p)||d(f(r,t,p,a)),l(r[t][p])})),v={},b={537:()=>h("default","@jupyterlab/coreutils",!1,[1,6,3,0]),869:()=>h("default","@jupyterlab/application",!1,[1,4,3,0]),176:()=>h("default","@jupyterlab/console",!1,[1,4,3,0]),918:()=>h("default","@jupyterlab/settingregistry",!1,[1,4,3,0]),954:()=>h("default","@jupyterlab/statedb",!1,[1,4,3,0]),528:()=>h("default","@jupyterlab/notebook",!1,[1,4,3,0]),916:()=>h("default","@jupyterlab/ui-components",!1,[1,4,3,0]),602:()=>h("default","@lumino/signaling",!1,[1,2,0,0]),256:()=>h("default","@lumino/widgets",!1,[1,2,3,1,,"alpha",0]),30:()=>h("default","@jupyterlab/apputils",!1,[1,4,4,0]),778:()=>h("default","@jupyterlab/services",!1,[1,7,3,0]),53:()=>h("default","@lumino/algorithm",!1,[1,2,0,0]),262:()=>h("default","@lumino/coreutils",!1,[1,2,0,0]),209:()=>h("default","@lumino/domutils",!1,[1,2,0,0]),355:()=>h("default","@lumino/dragdrop",!1,[1,2,0,0]),583:()=>h("default","@lumino/polling",!1,[1,2,0,0]),345:()=>h("default","react",!1,[1,18,2,0]),628:()=>h("default","react-dom",!1,[1,18,2,0])},g={296:[537,869,176,918,954,528,916,602,256,30,778,53,262,209,355,583,345,628]},m={},j.f.consumes=(e,r)=>{j.o(g,e)&&g[e].forEach((e=>{if(j.o(v,e))return r.push(v[e]);if(!m[e]){var t=r=>{v[e]=0,j.m[e]=t=>{delete j.c[e],t.exports=r()}};m[e]=!0;var n=r=>{delete v[e],j.m[e]=t=>{throw delete j.c[e],r}};try{var a=b[e]();a.then?r.push(v[e]=a.then(t).catch(n)):t(a)}catch(e){n(e)}}}))},(()=>{j.b=document.baseURI||self.location.href;var e={869:0};j.f.j=(r,t)=>{var n=j.o(e,r)?e[r]:void 0;if(0!==n)if(n)t.push(n[2]);else{var a=new Promise(((t,a)=>n=e[r]=[t,a]));t.push(n[2]=a);var o=j.p+j.u(r),i=new Error;j.l(o,(t=>{if(j.o(e,r)&&(0!==(n=e[r])&&(e[r]=void 0),n)){var a=t&&("load"===t.type?"missing":t.type),o=t&&t.target&&t.target.src;i.message="Loading chunk "+r+" failed.\n("+a+": "+o+")",i.name="ChunkLoadError",i.type=a,i.request=o,n[1](i)}}),"chunk-"+r,r)}};var r=(r,t)=>{var n,a,[o,i,l]=t,u=0;if(o.some((r=>0!==e[r]))){for(n in i)j.o(i,n)&&(j.m[n]=i[n]);l&&l(j)}for(r&&r(t);u<o.length;u++)a=o[u],j.o(e,a)&&e[a]&&e[a][0](),e[a]=0},t=self.webpackChunkipyparallel_labextension=self.webpackChunkipyparallel_labextension||[];t.forEach(r.bind(null,0)),t.push=r.bind(null,t.push.bind(t))})(),j.nc=void 0;var S=j(211);(_JUPYTERLAB=void 0===_JUPYTERLAB?{}:_JUPYTERLAB)["ipyparallel-labextension"]=S})();