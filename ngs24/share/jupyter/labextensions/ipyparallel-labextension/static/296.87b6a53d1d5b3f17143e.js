"use strict";(self.webpackChunkipyparallel_labextension=self.webpackChunkipyparallel_labextension||[]).push([[296],{296:(e,t,n)=>{n.r(t),n.d(t,{default:()=>O});var s=n(537),i=n(869),r=n(176),a=n(918),o=n(954),l=n(528),c=n(916),u=n(602),d=n(256),p=n(30),h=n(778),g=n(53),m=n(262),C=n(209),v=n(355),f=n(583),y=n(345);class w extends y.Component{constructor(e){let t;super(e),t=e.initialModel,this.state={model:t}}componentDidUpdate(){let e={...this.state.model};this.props.stateEscapeHatch(e)}onScaleChanged(e){this.setState({model:{...this.state.model,n:parseInt(e.target.value||null,null)}})}onProfileChanged(e){this.setState({model:{...this.state.model,profile:e.target.value}})}onClusterIdChanged(e){this.setState({model:{...this.state.model,cluster_id:e.target.value}})}render(){const e=this.state.model;return y.createElement("div",null,y.createElement("div",{className:"ipp-DialogSection"},y.createElement("div",{className:"ipp-DialogSection-item"},y.createElement("span",{className:"ipp-DialogSection-label"},"Profile"),y.createElement("input",{className:"ipp-DialogInput",value:e.profile,type:"string",placeholder:"default",onChange:e=>{this.onProfileChanged(e)}})),y.createElement("div",{className:"ipp-DialogSection-item"},y.createElement("span",{className:"ipp-DialogSection-label"},"Cluster ID"),y.createElement("input",{className:"ipp-DialogInput",value:e.cluster_id,type:"string",placeholder:"auto",onChange:e=>{this.onClusterIdChanged(e)}})),y.createElement("div",{className:"ipp-DialogSection-item"},y.createElement("span",{className:"ipp-DialogSection-label"},"Engines"),y.createElement("input",{className:"ipp-DialogInput",value:e.n,type:"number",step:"1",placeholder:"auto",onChange:e=>{this.onScaleChanged(e)}}))))}}var b,_=n(338);!function(e){e.injectClientCode="ipyparallel:inject-client-code",e.newCluster="ipyparallel:new-cluster",e.startCluster="ipyparallel:start-cluster",e.stopCluster="ipyparallel:stop-cluster",e.scaleCluster="ipyparallel:scale-cluster",e.toggleAutoStartClient="ipyparallel:toggle-auto-start-client"}(b||(b={}));const I="ipyparallel/clusters";class L extends d.Widget{constructor(e){super(),this._dragData=null,this._clusters=[],this._activeClusterChanged=new u.Signal(this),this._serverErrorShown=!1,this._isReady=!0,this.addClass("ipp-ClusterManager"),this._serverSettings=h.ServerConnection.makeSettings(),this._injectClientCodeForCluster=e.injectClientCodeForCluster,this._getClientCodeForCluster=e.getClientCodeForCluster,this._registry=e.registry,this._setActiveById=e=>{const t=this._clusters.find((t=>t.id===e));if(!t)return;const n=this._activeCluster;n&&n.id===t.id||(this._activeCluster=t,this._activeClusterChanged.emit({name:"cluster",oldValue:n,newValue:t}),this.update())};const t=this.layout=new d.PanelLayout;this._clusterListing=new d.Widget,this._clusterListing.addClass("ipp-ClusterListing");const n=new p.Toolbar,s=new d.Widget;s.node.textContent="CLUSTERS",s.addClass("ipp-ClusterManager-label"),n.addItem("label",s),n.addItem("refresh",new p.ToolbarButton({icon:c.refreshIcon,onClick:async()=>this._updateClusterList(),tooltip:"Refresh Cluster List"})),n.addItem(b.newCluster,new p.CommandToolbarButton({commands:this._registry,id:b.newCluster})),t.addWidget(n),t.addWidget(this._clusterListing),this._updateClusterList(),this._poll=new f.Poll({factory:async()=>{await this._updateClusterList()},frequency:{interval:5e3,backoff:!0,max:6e4},standby:"when-hidden"})}get activeCluster(){return this._activeCluster}setActiveCluster(e){this._setActiveById(e)}get activeClusterChanged(){return this._activeClusterChanged}get isReady(){return this._isReady}get clusters(){return this._clusters}async refresh(){await this._updateClusterList()}async create(){const e=await function(e){let t={...e};return(0,p.showDialog)({title:"New Cluster",body:y.createElement(w,{initialModel:e,stateEscapeHatch:e=>{t=e}}),buttons:[p.Dialog.cancelButton(),p.Dialog.okButton({label:"CREATE"})]}).then((e=>e.button.accept?t:null))}({});if(e)return await this._newCluster(e)}async start(e){if(!this._clusters.find((t=>t.id===e)))throw Error(`Cannot find cluster ${e}`);await this._startById(e)}async stop(e){if(!this._clusters.find((t=>t.id===e)))throw Error(`Cannot find cluster ${e}`);await this._stopById(e)}async scale(e){if(!this._clusters.find((t=>t.id===e)))throw Error(`Cannot find cluster ${e}`);return await this._scaleById(e)}dispose(){this.isDisposed||(this._poll.dispose(),super.dispose())}onUpdateRequest(e){this.isVisible&&(0,_.H)(this._clusterListing.node).render(y.createElement(x,{clusters:this._clusters,activeClusterId:this._activeCluster&&this._activeCluster.id||"",scaleById:e=>this._scaleById(e),startById:e=>this._startById(e),stopById:e=>this._stopById(e),setActiveById:this._setActiveById,injectClientCodeForCluster:this._injectClientCodeForCluster}))}onAfterShow(e){this.update()}onAfterAttach(e){super.onAfterAttach(e);let t=this._clusterListing.node;t.addEventListener("p-dragenter",this),t.addEventListener("p-dragleave",this),t.addEventListener("p-dragover",this),t.addEventListener("mousedown",this)}onBeforeDetach(e){let t=this._clusterListing.node;t.removeEventListener("p-dragenter",this),t.removeEventListener("p-dragleave",this),t.removeEventListener("p-dragover",this),t.removeEventListener("mousedown",this),document.removeEventListener("mouseup",this,!0),document.removeEventListener("mousemove",this,!0)}handleEvent(e){switch(e.type){case"mousedown":this._evtMouseDown(e);break;case"mouseup":this._evtMouseUp(e);break;case"mousemove":this._evtMouseMove(e)}}_evtMouseDown(e){const{button:t,shiftKey:n}=e;if(0!==t&&2!==t)return;if(n&&2===t)return;const s=this._findCluster(e);-1!==s&&(this._dragData={pressX:e.clientX,pressY:e.clientY,index:s},document.addEventListener("mouseup",this,!0),document.addEventListener("mousemove",this,!0),e.preventDefault())}_evtMouseUp(e){0===e.button&&this._drag||(document.removeEventListener("mousemove",this,!0),document.removeEventListener("mouseup",this,!0)),e.preventDefault(),e.stopPropagation()}_evtMouseMove(e){let t=this._dragData;if(!t)return;let n=Math.abs(e.clientX-t.pressX),s=Math.abs(e.clientY-t.pressY);(n>=5||s>=5)&&(e.preventDefault(),e.stopPropagation(),this._startDrag(t.index,e.clientX,e.clientY))}async _startDrag(e,t,n){const s=this._clusters[e],i=this._clusterListing.node.querySelector(`li.ipp-ClusterListingItem[data-cluster-id="${s.id}"]`),r=S.createDragImage(i);this._drag=new v.Drag({mimeData:new m.MimeData,dragImage:r,supportedActions:"copy",proposedAction:"copy",source:this});const a=this._getClientCodeForCluster(s);this._drag.mimeData.setData("text/plain",a);const o=[{cell_type:"code",source:a,outputs:[],execution_count:null,metadata:{}}];return this._drag.mimeData.setData("application/vnd.jupyter.cells",o),document.removeEventListener("mousemove",this,!0),document.removeEventListener("mouseup",this,!0),this._drag.start(t,n).then((e=>{this.isDisposed||(this._drag=null,this._dragData=null)}))}async _newCluster(e){this._isReady=!1,this._registry.notifyCommandChanged(b.newCluster);const t=await h.ServerConnection.makeRequest(`${this._serverSettings.baseUrl}${I}`,{method:"POST",body:JSON.stringify(e)},this._serverSettings);if(200!==t.status){const e=await t.json();throw(0,p.showErrorMessage)("Cluster Create Error",e),this._isReady=!0,this._registry.notifyCommandChanged(b.newCluster),e}const n=await t.json();return await this._updateClusterList(),this._isReady=!0,this._registry.notifyCommandChanged(b.newCluster),n}async _updateClusterList(){const e=await h.ServerConnection.makeRequest(`${this._serverSettings.baseUrl}${I}`,{},this._serverSettings);if(200!==e.status){const e=new Error("Failed to list clusters: might the server extension not be installed/enabled?");throw this._serverErrorShown||((0,p.showErrorMessage)("IPP Extension Server Error",e),this._serverErrorShown=!0),e}const t=await e.json();if(this._clusters=t,!this._clusters.find((e=>e.id===(this._activeCluster&&this._activeCluster.id)))){const e=this._clusters[0]&&this._clusters[0].id||"";this._setActiveById(e)}this.update()}async _startById(e){const t=await h.ServerConnection.makeRequest(`${this._serverSettings.baseUrl}${I}/${e}`,{method:"POST"},this._serverSettings);if(t.status>299){const e=await t.json();throw(0,p.showErrorMessage)("Failed to start cluster",e),e}await this._updateClusterList()}async _stopById(e){const t=await h.ServerConnection.makeRequest(`${this._serverSettings.baseUrl}${I}/${e}`,{method:"DELETE"},this._serverSettings);if(204!==t.status){const e=await t.json();throw(0,p.showErrorMessage)("Failed to close cluster",e),e}await this._updateClusterList()}async _scaleById(e){const t=this._clusters.find((t=>t.id===e));if(!t)throw Error(`Failed to find cluster ${e} to scale`);(0,p.showErrorMessage)("Scale not implemented","");const n=t;if(m.JSONExt.deepEqual(n,t))return Promise.resolve(t);const s=await h.ServerConnection.makeRequest(`${this._serverSettings.baseUrl}${I}/${e}`,{method:"PATCH",body:JSON.stringify(n)},this._serverSettings);if(200!==s.status){const e=await s.json();throw(0,p.showErrorMessage)("Failed to scale cluster",e),e}const i=await s.json();return await this._updateClusterList(),i}_findCluster(e){const t=Array.from(this.node.querySelectorAll("li.ipp-ClusterListingItem"));return g.ArrayExt.findFirstIndex(t,(t=>C.ElementExt.hitTest(t,e.clientX,e.clientY)))}}function x(e){let t=e.clusters.map((t=>y.createElement(E,{isActive:t.id===e.activeClusterId,key:t.id,cluster:t,scale:()=>e.scaleById(t.id),start:()=>e.startById(t.id),stop:()=>e.stopById(t.id),setActive:()=>e.setActiveById(t.id),injectClientCode:()=>e.injectClientCodeForCluster(t)})));return y.createElement("div",null,y.createElement("ul",{className:"ipp-ClusterListing-list"},t))}function E(e){const{cluster:t,isActive:n,setActive:s,scale:i,start:r,stop:a,injectClientCode:o}=e;let l="ipp-ClusterListingItem";l=n?`${l} jp-mod-active`:l;let c="Stopped";t.controller&&(c=t.controller.state.state,"after"==c&&(c="Stopped"));let u="Stopped"===c?"DELETE":"STOP";return y.createElement("li",{className:l,"data-cluster-id":t.id,onClick:e=>{s(),e.stopPropagation()}},y.createElement("div",{className:"ipp-ClusterListingItem-title"},t.id),y.createElement("div",{className:"ipp-ClusterListingItem-stats"},"State: ",c),y.createElement("div",{className:"ipp-ClusterListingItem-stats"},"Number of engines: ",t.engines.n||t.cluster.n||"auto"),y.createElement("div",{className:"ipp-ClusterListingItem-button-panel"},y.createElement("button",{className:"ipp-ClusterListingItem-button ipp-ClusterListingItem-code ipp-CodeIcon jp-mod-styled",onClick:e=>{o(),e.stopPropagation()},title:`Inject client code for ${t.id}`}),y.createElement("button",{className:"ipp-ClusterListingItem-button ipp-ClusterListingItem-start jp-mod-styled "+("Stopped"==c?"":"ipp-hidden"),onClick:async e=>(e.stopPropagation(),r()),title:`Start ${t.id}`},"START"),y.createElement("button",{className:"ipp-ClusterListingItem-button ipp-ClusterListingItem-scale jp-mod-styled ipp-hidden",onClick:async e=>(e.stopPropagation(),i()),title:`Rescale ${t.id}`},"SCALE"),y.createElement("button",{className:"ipp-ClusterListingItem-button ipp-ClusterListingItem-stop jp-mod-styled "+("Stopped"===c&&""===t.cluster.cluster_id?"ipp-hidden":""),onClick:async e=>(e.stopPropagation(),a()),title:u},u)))}var S;!function(e){e.createDragImage=function(e){const t=e.cloneNode(!0);return t.classList.add("ipp-ClusterListingItem-drag"),t}}(S||(S={}));class j extends d.Widget{constructor(e){super(),this.addClass("ipp-Sidebar");let t=this.layout=new d.PanelLayout;const n=e.clientCodeInjector,s=e.clientCodeGetter;this._clusters=new L({registry:e.registry,injectClientCodeForCluster:n,getClientCodeForCluster:s}),t.addWidget(this._clusters)}get clusterManager(){return this._clusters}}var k=n(72),M=n.n(k),D=n(825),A=n.n(D),N=n(659),P=n.n(N),B=n(56),R=n.n(B),T=n(540),$=n.n(T),F=n(113),W=n.n(F),U=n(466),z={};z.styleTagTransform=W(),z.setAttributes=R(),z.insert=P().bind(null,"head"),z.domAPI=A(),z.insertStyleElement=$(),M()(U.A,z),U.A&&U.A.locals&&U.A.locals;const q="ipyparallel-labextension:plugin",H="IPython Parallel",O={activate:async function(e,t,n,i,r,a){const o="ipp-cluster-launcher",l=!!a,d="tree"==s.PageConfig.getOption("retroPage"),p=async s=>{const i=await V.getCurrentEditor(e,n,t);i&&V.injectClientCode(s,i)},h=new j({clientCodeInjector:p,clientCodeGetter:V.getClientCode,registry:e.commands});if(h.id=o,h.title.icon=new c.LabIcon({name:"ipyparallel:logo",svgstr:'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<svg\n   xmlns="http://www.w3.org/2000/svg"\n   version="1.1"\n   viewBox="0 0 20 20"\n   height="20"\n   width="20">\n   \x3c!-- text: IP in Source Code Pro --\x3e\n    <g\n       aria-label="IP">\n      <path\n         class="jp-icon3 jp-icon-selectable"\n         fill="#616161"\n         d="m 1.619125,15.248 v -1.136 h 2.608 V 5.8720001 h -2.608 v -1.12 h 6.56 v 1.12 h -2.608 V 14.112 h 2.608 v 1.136 z" />\n      <path\n         class="jp-icon3 jp-icon-selectable"\n         fill="#616161"\n         d="M 11.324875,15.248 V 4.7520001 h 3.168 q 1.168,0 2.032,0.288 0.88,0.288 1.36,0.976 0.496,0.672 0.496,1.824 0,1.104 -0.496,1.824 -0.48,0.7199999 -1.36,1.0719999 -0.88,0.352 -2.032,0.352 h -1.84 v 4.16 z m 1.328,-5.248 h 1.68 q 1.376,0 2.032,-0.5119999 0.672,-0.528 0.672,-1.648 0,-1.136 -0.672,-1.568 -0.672,-0.448 -2.032,-0.448 h -1.68 z" />\n  </g>\n</svg>\n'}),l)a.add(h,"left",{rank:200}),h.title.caption=H;else if(d){const t=e.shell.currentWidget;t.addWidget(h),t.tabBar.addTab(h.title),h.title.label=H}h.clusterManager.activeClusterChanged.connect((async()=>{const e=h.clusterManager.activeCluster;return r.save(o,{cluster:e?e.id:""})}));const g=async e=>{if(!e)return;const t=h.clusterManager.activeCluster;return t&&await V.shouldUseKernel(e.kernel)?V.createClientForKernel(t,e.kernel):void 0},m=[n,t],C=async e=>{if(e.session&&e.session.kernel&&"restarting"===e.session.kernel.status)return g(e.session)},v=(e,t)=>{t.sessionContext.statusChanged.connect(C)},f=()=>{m.forEach((e=>{e.forEach((async e=>{const t=e.sessionContext.session;if(t&&await V.shouldUseKernel(t.kernel))return g(t)}))}))};Promise.all([i.load(q),r.fetch(o)]).then((async e=>{const t=e[0];if(!t)return void console.warn("Unable to retrieve ipp-labextension settings");const n=e[1],s=n?n.cluster:"",i=()=>{u.Signal.clearData(v),u.Signal.clearData(C),u.Signal.clearData(f)};i(),t.changed.connect(i),s&&(await h.clusterManager.refresh(),h.clusterManager.setActiveCluster(s))})),e.commands.addCommand(b.injectClientCode,{label:"Inject IPython Client Connection Code",execute:async()=>{const t=V.clusterFromClick(e,h.clusterManager);if(t)return await p(t)}}),e.commands.addCommand(b.newCluster,{label:e=>e.isPalette?"Create New Cluster":"NEW",execute:()=>h.clusterManager.create(),iconClass:e=>e.isPalette?"":"jp-AddIcon jp-Icon jp-Icon-16",isEnabled:()=>h.clusterManager.isReady,caption:()=>h.clusterManager.isReady?"Start New Cluster":"Cluster starting..."}),e.commands.addCommand(b.startCluster,{label:"Start Cluster",execute:()=>{const t=V.clusterFromClick(e,h.clusterManager);if(t)return h.clusterManager.start(t.id)}}),e.commands.addCommand(b.stopCluster,{label:"Shutdown Cluster",execute:()=>{const t=V.clusterFromClick(e,h.clusterManager);if(t)return h.clusterManager.stop(t.id)}}),e.commands.addCommand(b.scaleCluster,{label:"Scale Cluster…",execute:()=>{const t=V.clusterFromClick(e,h.clusterManager);if(t)return h.clusterManager.scale(t.id)}}),e.contextMenu.addItem({command:b.injectClientCode,selector:".ipp-ClusterListingItem",rank:10}),e.contextMenu.addItem({command:b.stopCluster,selector:".ipp-ClusterListingItem",rank:3}),e.contextMenu.addItem({command:b.scaleCluster,selector:".ipp-ClusterListingItem",rank:2}),e.contextMenu.addItem({command:b.startCluster,selector:".ipp-ClusterListing-list",rank:1})},id:q,requires:[r.IConsoleTracker,l.INotebookTracker,a.ISettingRegistry,o.IStateDB],optional:[i.ILabShell],autoStart:!0};var V;!function(e){function t(e){return`import ipyparallel as ipp\n\ncluster = ipp.Cluster.from_file("${e.cluster_file}")\nrc = cluster.connect_client_sync()\nrc`}e.id=0,e.shouldUseKernel=async function(e){if(!e)return!1;const t=await e.spec;return!!t&&-1!==t.language.toLowerCase().indexOf("python")},e.createClientForKernel=async function(e,n){const s={store_history:!1,code:t(e)};return new Promise(((e,t)=>{n.requestExecute(s).onIOPub=t=>{"display_data"===t.header.msg_type&&e(void 0)}}))},e.injectClientCode=function(e,n){const s=n.getCursorPosition(),i=n.getOffsetAt(s),r=t(e);n.model.sharedModel.updateSource(i,i,r)},e.getClientCode=t,e.getCurrentKernel=function(e,t,n){var s,i,r,a;let o,l=e.currentWidget;return l&&t.has(l)?o=null===(s=l.sessionContext.session)||void 0===s?void 0:s.kernel:l&&n.has(l)?o=null===(i=l.sessionContext.session)||void 0===i?void 0:i.kernel:t.currentWidget?o=null===(r=t.currentWidget.sessionContext.session)||void 0===r?void 0:r.kernel:n.currentWidget&&(o=null===(a=n.currentWidget.sessionContext.session)||void 0===a?void 0:a.kernel),o},e.getCurrentEditor=async function(e,t,n){let s,i=e.shell.currentWidget;if(i&&t.has(i)){l.NotebookActions.insertAbove(i.content);const e=i.content.activeCell;await e.ready,s=e&&e.editor}else if(i&&n.has(i)){const e=i.console.promptCell;await e.ready,s=e&&e.editor}else if(t.currentWidget){const e=t.currentWidget;l.NotebookActions.insertAbove(e.content);const n=e.content.activeCell;await n.ready,s=n&&n.editor}else if(n.currentWidget){const e=n.currentWidget.console.promptCell;await e.ready,s=e&&e.editor}return s},e.clusterFromClick=function(e,t){const n=e.contextMenuHitTest((e=>!!e.dataset.clusterId));if(!n)return;const s=n.dataset.clusterId;return t.clusters.find((e=>e.id===s))}}(V||(V={}))},466:(e,t,n)=>{n.d(t,{A:()=>g});var s=n(601),i=n.n(s),r=n(314),a=n.n(r),o=n(417),l=n.n(o),c=new URL(n(724),n.b),u=new URL(n(620),n.b),d=a()(i()),p=l()(c),h=l()(u);d.push([e.id,`:root {\n  --ipp-launch-button-height: 24px;\n}\n\n/**\n * Rules related to the overall sidebar panel.\n */\n\n.ipp-Sidebar {\n  background: var(--jp-layout-color1);\n  color: var(--jp-ui-font-color1);\n  font-size: var(--jp-ui-font-size1);\n  overflow: auto;\n}\n\n/**\n * Rules related to the cluster manager.\n */\n\n.ipp-ClusterManager .jp-Toolbar {\n  align-items: center;\n}\n\n.ipp-ClusterManager .jp-Toolbar .ipp-ClusterManager-label {\n  flex: 0 0 auto;\n  font-weight: 600;\n  text-transform: uppercase;\n  letter-spacing: 1px;\n  font-size: var(--jp-ui-font-size0);\n  padding: 8px 8px 8px 12px;\n  margin: 0px;\n}\n\n.ipp-ClusterManager button.jp-Button > span {\n  display: flex;\n  flex-direction: row;\n  align-items: center;\n}\n\n.ipp-ClusterListing ul.ipp-ClusterListing-list {\n  list-style-type: none;\n  padding: 0;\n  margin: 0;\n}\n\n.ipp-ClusterListingItem {\n  display: inline-block;\n  list-style-type: none;\n  padding: 8px;\n  width: 100%;\n  white-space: nowrap;\n  overflow: hidden;\n  text-overflow: ellipsis;\n  cursor: grab;\n}\n\n.ipp-ClusterListingItem-drag {\n  opacity: 0.7;\n  color: var(--jp-ui-font-color1);\n  cursor: grabbing;\n  max-width: 260px;\n  transform: translateX(-50%) translateY(-50%);\n}\n\n.ipp-ClusterListingItem-title {\n  margin: 0px;\n  font-size: var(--jp-ui-font-size2);\n}\n\n.ipp-ClusterListingItem-link a {\n  text-decoration: none;\n  color: var(--jp-content-link-color);\n}\n\n.ipp-ClusterListingItem-link a:hover {\n  text-decoration: underline;\n}\n\n.ipp-ClusterListingItem-link a:visited {\n  color: var(--jp-content-link-color);\n}\n\n.ipp-ClusterListingItem:hover {\n  background: var(--jp-layout-color2);\n}\n\n.ipp-ClusterListingItem.jp-mod-active {\n  color: white;\n  background: var(--jp-brand-color0);\n}\n\n.ipp-ClusterListingItem.jp-mod-active a,\n.ipp-ClusterListingItem.jp-mod-active a:visited {\n  color: white;\n}\n\n.ipp-ClusterListingItem button.jp-mod-styled {\n  background-color: transparent;\n}\n\n.ipp-ClusterListingItem button.jp-mod-styled:hover {\n  background-color: var(--jp-layout-color3);\n}\n\n.ipp-ClusterListingItem.jp-mod-active button.jp-mod-styled:hover {\n  background-color: var(--jp-brand-color1);\n}\n\n.ipp-ClusterListingItem-button-panel {\n  display: flex;\n  align-content: center;\n}\n\nbutton.ipp-ClusterListingItem-stop {\n  color: var(--jp-warn-color1);\n  font-weight: 600;\n}\n\nbutton.ipp-ClusterListingItem-scale {\n  color: var(--jp-accent-color1);\n  font-weight: 600;\n}\n\nbutton.ipp-ClusterListingItem-start {\n  color: var(--jp-accent-color1);\n  font-weight: 600;\n}\n\nbutton.ipp-hidden {\n  display: none;\n}\n\n.ipp-ClusterListingItem button.ipp-ClusterListingItem-code.jp-mod-styled {\n  margin: 0 4px 0 4px;\n  background-repeat: no-repeat;\n  background-position: center;\n}\n\n/**\n * Rules for the scaling dialog.\n */\n\n.ipp-DialogHeader {\n  font-size: var(--jp-ui-font-size2);\n}\n\n.ipp-DialogSection {\n  margin-left: 24px;\n}\n\n.ipp-DialogSection-item {\n  display: flex;\n  align-items: center;\n  justify-content: space-around;\n  margin: 12px 0 12px 0;\n}\n\n.ipp-DialogHeader input[type="checkbox"] {\n  position: relative;\n  top: 4px;\n  left: 4px;\n  margin: 0 0 0 8px;\n}\n\n.ipp-DialogSection input[type="number"] {\n  width: 72px;\n}\n\n.ipp-DialogSection-label.ipp-mod-disabled {\n  color: var(--jp-ui-font-color3);\n}\n\n.ipp-DialogSection input[type="number"]:disabled {\n  color: var(--jp-ui-font-color3);\n}\n\n/**\n * Rules for the logos.\n */\n\n.ipp-SearchIcon {\n  background-image: var(--jp-icon-search);\n}\n\n[data-jp-theme-light="true"] .ipp-CodeIcon {\n  background-image: url(${p});\n}\n\n[data-jp-theme-light="false"] .ipp-CodeIcon {\n  background-image: url(${h});\n}\n\n.ipp-ClusterListingItem.jp-mod-active .ipp-CodeIcon {\n  background-image: url(${h});\n}\n`,""]);const g=d},314:e=>{e.exports=function(e){var t=[];return t.toString=function(){return this.map((function(t){var n="",s=void 0!==t[5];return t[4]&&(n+="@supports (".concat(t[4],") {")),t[2]&&(n+="@media ".concat(t[2]," {")),s&&(n+="@layer".concat(t[5].length>0?" ".concat(t[5]):""," {")),n+=e(t),s&&(n+="}"),t[2]&&(n+="}"),t[4]&&(n+="}"),n})).join("")},t.i=function(e,n,s,i,r){"string"==typeof e&&(e=[[null,e,void 0]]);var a={};if(s)for(var o=0;o<this.length;o++){var l=this[o][0];null!=l&&(a[l]=!0)}for(var c=0;c<e.length;c++){var u=[].concat(e[c]);s&&a[u[0]]||(void 0!==r&&(void 0===u[5]||(u[1]="@layer".concat(u[5].length>0?" ".concat(u[5]):""," {").concat(u[1],"}")),u[5]=r),n&&(u[2]?(u[1]="@media ".concat(u[2]," {").concat(u[1],"}"),u[2]=n):u[2]=n),i&&(u[4]?(u[1]="@supports (".concat(u[4],") {").concat(u[1],"}"),u[4]=i):u[4]="".concat(i)),t.push(u))}},t}},417:e=>{e.exports=function(e,t){return t||(t={}),e?(e=String(e.__esModule?e.default:e),/^['"].*['"]$/.test(e)&&(e=e.slice(1,-1)),t.hash&&(e+=t.hash),/["'() \t\n]|(%20)/.test(e)||t.needQuotes?'"'.concat(e.replace(/"/g,'\\"').replace(/\n/g,"\\n"),'"'):e):e}},601:e=>{e.exports=function(e){return e[1]}},338:(e,t,n)=>{var s=n(628);t.H=s.createRoot,s.hydrateRoot},72:e=>{var t=[];function n(e){for(var n=-1,s=0;s<t.length;s++)if(t[s].identifier===e){n=s;break}return n}function s(e,s){for(var r={},a=[],o=0;o<e.length;o++){var l=e[o],c=s.base?l[0]+s.base:l[0],u=r[c]||0,d="".concat(c," ").concat(u);r[c]=u+1;var p=n(d),h={css:l[1],media:l[2],sourceMap:l[3],supports:l[4],layer:l[5]};if(-1!==p)t[p].references++,t[p].updater(h);else{var g=i(h,s);s.byIndex=o,t.splice(o,0,{identifier:d,updater:g,references:1})}a.push(d)}return a}function i(e,t){var n=t.domAPI(t);return n.update(e),function(t){if(t){if(t.css===e.css&&t.media===e.media&&t.sourceMap===e.sourceMap&&t.supports===e.supports&&t.layer===e.layer)return;n.update(e=t)}else n.remove()}}e.exports=function(e,i){var r=s(e=e||[],i=i||{});return function(e){e=e||[];for(var a=0;a<r.length;a++){var o=n(r[a]);t[o].references--}for(var l=s(e,i),c=0;c<r.length;c++){var u=n(r[c]);0===t[u].references&&(t[u].updater(),t.splice(u,1))}r=l}}},659:e=>{var t={};e.exports=function(e,n){var s=function(e){if(void 0===t[e]){var n=document.querySelector(e);if(window.HTMLIFrameElement&&n instanceof window.HTMLIFrameElement)try{n=n.contentDocument.head}catch(e){n=null}t[e]=n}return t[e]}(e);if(!s)throw new Error("Couldn't find a style target. This probably means that the value for the 'insert' parameter is invalid.");s.appendChild(n)}},540:e=>{e.exports=function(e){var t=document.createElement("style");return e.setAttributes(t,e.attributes),e.insert(t,e.options),t}},56:(e,t,n)=>{e.exports=function(e){var t=n.nc;t&&e.setAttribute("nonce",t)}},825:e=>{e.exports=function(e){if("undefined"==typeof document)return{update:function(){},remove:function(){}};var t=e.insertStyleElement(e);return{update:function(n){!function(e,t,n){var s="";n.supports&&(s+="@supports (".concat(n.supports,") {")),n.media&&(s+="@media ".concat(n.media," {"));var i=void 0!==n.layer;i&&(s+="@layer".concat(n.layer.length>0?" ".concat(n.layer):""," {")),s+=n.css,i&&(s+="}"),n.media&&(s+="}"),n.supports&&(s+="}");var r=n.sourceMap;r&&"undefined"!=typeof btoa&&(s+="\n/*# sourceMappingURL=data:application/json;base64,".concat(btoa(unescape(encodeURIComponent(JSON.stringify(r))))," */")),t.styleTagTransform(s,e,t.options)}(t,e,n)},remove:function(){!function(e){if(null===e.parentNode)return!1;e.parentNode.removeChild(e)}(t)}}}},113:e=>{e.exports=function(e,t){if(t.styleSheet)t.styleSheet.cssText=e;else{for(;t.firstChild;)t.removeChild(t.firstChild);t.appendChild(document.createTextNode(e))}}},620:e=>{e.exports="data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='%23E0E0E0' width='24' height='24' viewBox='0 0 24 24'%3e%3cpath fill='none' d='M0 0h24v24H0V0z'/%3e%3cpath d='M9.4 16.6L4.8 12l4.6-4.6L8 6l-6 6 6 6 1.4-1.4zm5.2 0l4.6-4.6-4.6-4.6L16 6l6 6-6 6-1.4-1.4z'/%3e%3c/svg%3e"},724:e=>{e.exports="data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='%23616161' width='24' height='24' viewBox='0 0 24 24'%3e%3cpath fill='none' d='M0 0h24v24H0V0z'/%3e%3cpath d='M9.4 16.6L4.8 12l4.6-4.6L8 6l-6 6 6 6 1.4-1.4zm5.2 0l4.6-4.6-4.6-4.6L16 6l6 6-6 6-1.4-1.4z'/%3e%3c/svg%3e"}}]);